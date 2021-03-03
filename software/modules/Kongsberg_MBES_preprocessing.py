"""Kongsberg_MBES_preporocessing.

Module which preprocesses raw Kongsberg EM series bathymetry and backscatter
from ALL files.

The module is based on the pyall reader. For the bathymetry preprocessing, the
XYZ and position datgrams are used. Based on the processed bathymetry, the
backscatter time series in the seabed image datagram is georeferenced.

This module contains the following functions:

    * decode_raw_bathymetry - Decodes XYZ and positions.
    * process_navigation - Interpolates positions to XYZ ping time.
    * locate_soundings - Transforms the XYZ vessel coordinates to geographic.
    * decode_raw_backscatter - Georeferences the backscatter.
    * UTM_EPSG - Helper function to fin the UTM zone of geographic coordinates.
"""

import pandas as pd
import numpy as np
import math
import pyall
from functools import reduce
from geographiclib.geodesic import Geodesic


def decode_raw_bathymetry(filename):
    """Decode raw Kongsberg ALL bathymetry and return dataframes.

    Parameters
    ----------
    filename : str
        The file location of the Kongsberg ALL file.

    Returns
    -------
    position : pandas DataFrame
        Position datagrams of the ALL file.
    XYZ : pandas DataFrame
        XYZ datagrams of the ALL file.
    """
    position = []
    XYZ = []

    data = pyall.ALLReader(filename)

    while data.moreData():
        typeOfDatagram, datagram = data.readDatagram()
        # Read XYZ datagram
        if typeOfDatagram == 'X':
            datagram.read()
            # Append relevant information to XYZ list
            XYZ.append(
                {
                    "date_time": pyall.to_DateTime(
                        datagram.RecordDate,
                        datagram.Time
                        ),
                    "ping_counter": datagram.Counter,
                    "heading": datagram.Heading,
                    "transducer_depth": datagram.TransducerDepth,
                    "across_track_distance": datagram.AcrossTrackDistance,
                    "along_track_distance": datagram.AlongTrackDistance,
                    "depth": datagram.Depth,
                    "reflectivity": datagram.Reflectivity,
                    "classification": datagram.RealtimeCleaningInformation,
                    "beam_index": list(range(datagram.NBeams))
                }
            )
            continue

        # Read position datagram
        if typeOfDatagram == 'P':
            datagram.read()
            # Append relevant information to position list
            position.append(
                {
                    "date_time": pyall.to_DateTime(
                        datagram.RecordDate,
                        datagram.Time
                        ),
                    "latitude": datagram.Latitude,
                    "longitude": datagram.Longitude,
                    "descriptor": datagram.Descriptor
                }
            )
            continue

    # Convert lists of dictionaries to DataFrames with datetimeindex
    position = pd.DataFrame(position).set_index(['date_time'])
    print(f'Decoded {len(position)} position datagram(s).')
    XYZ = pd.DataFrame(XYZ).set_index(['date_time'])
    print(f'Decoded {len(XYZ)} XYZ datagram(s).')
    return position, XYZ


def process_navigation(position, XYZ, method='values', limit=2, time_lag=0.0):
    """Interpolate navigation data to ping time.

    Parameters
    ----------
    position : pandas DataFrame
        Position datagrams of the ALL file.
    XYZ : pandas DataFrame
        XYZ datagrams of the ALL file.
    method : str (default: 'values')
        Interpolation method.
    limit: int (default: 2)
        Number of consecutive pings to interpolate.
    time_lag: float (default: 0.0)
        Time lag [sec] to add to navigation.

    Returns
    -------
    XYZ_positions : pandas DataFrame
        XYZ data with associated positions.
    """
    # Filter positions from active position system only
    position.where(position['descriptor'] > 128, inplace=True)
    position.dropna(inplace=True)

    # Add optional time lag
    if time_lag:
        position.index = position.index + pd.offsets.Second(time_lag)

    # Concatenate depths and positions
    XYZ_positions = pd.concat([XYZ, position], join='outer')
    XYZ_positions.drop(columns=['descriptor'], inplace=True)
    XYZ_positions.sort_index(inplace=True)

    # Interpolate latitude and longitude
    XYZ_positions['latitude'].interpolate(
        method=method,
        limit=limit,
        limit_direction='both',
        inplace=True
        )
    XYZ_positions['longitude'].interpolate(
        method=method,
        limit=limit,
        limit_direction='both',
        inplace=True
        )

    # Count number of pings without navigation information
    not_interpolated = XYZ_positions['latitude'].isnull().sum()
    print(f'{not_interpolated} ping(s) could not be positioned.')

    # Drop positions which lack required information
    XYZ_positions.dropna(subset=['ping_counter', 'latitude'], inplace=True)

    return XYZ_positions


def locate_soundings(XYZ_positions):
    """Georeference soundings.

    Parameters
    ----------
    XYZ_positions : pandas DataFrame
        XYZ data with associated positions.

    Returns
    -------
    XYZ_positions : pandas DataFrame
        Modified XYZ data with located soundings.
    """
    # Define geoid for direct geodetic problem
    GEOID = Geodesic.WGS84

    # Expand beam information lists
    beam_information = [
        'across_track_distance',
        'along_track_distance',
        'depth',
        'beam_index',
        'reflectivity',
        'classification'
        ]
    XYZ_positions = XYZ_positions.apply(
        lambda x: x.explode() if x.name in beam_information else x
        )

    def merge_vincenty(x, y, z, tx_depth, hdng, lat, lon):
        """Determine sounding positions using the Vincenty's formula."""
        azimuth = hdng + math.degrees(np.arctan2(y, x))
        distance = math.sqrt(x*x + y*y)
        geodesic = GEOID.Direct(lat, lon, azimuth, distance)
        depth = -(z + tx_depth)
        return geodesic['lat2'], geodesic['lon2'], depth

    # Apply Vincenty's formula to every beam in file
    XYZ_positions['latitude'],\
        XYZ_positions['longitude'],\
        XYZ_positions['depth'] = np.vectorize(merge_vincenty)(
            XYZ_positions.along_track_distance,
            XYZ_positions.across_track_distance,
            XYZ_positions.depth,
            XYZ_positions.transducer_depth,
            XYZ_positions.heading,
            XYZ_positions.latitude,
            XYZ_positions.longitude
            )

    # Drop columns which are not required anymore
    XYZ_positions.drop(
            columns=[
                'along_track_distance',
                'across_track_distance',
                'transducer_depth',
                'heading'
            ],
            inplace=True
            )

    XYZ_positions.set_index(['ping_counter', 'beam_index'], inplace=True)
    return XYZ_positions


class GeoReferencingAccumulator:
    """Accumulator for backscatter swath.

    Attributes
    ----------
    bathymetryOfSwath : pandas DataFrame
        Modified XYZ data with located soundings.
    georeferencedSamples: numpy array
        Array including the georeferenced samples.
    lastBeam: pyall cBeam instance
        Last beam considered in swath.
    nextBeam: pyall cBeam instance
        Next beam considered in swath.

    Methods
    -------
    update(nextBeam)
        Georeferences samples between beams.
    """

    def __init__(self, bathymetryOfSwath):
        """Init GeoReferencingAccumulator.

        Parameters
        ----------
        bathymetryOfSwath : pandas DataFrame
            Modified XYZ data with located soundings.
        """
        self.bathymetryOfSwath = bathymetryOfSwath
        self.georeferencedSamples = None
        self.lastBeam = None
        self.last_beam_index = None

    def update(self, nextBeam):
        """Georeference samples between beams.

        Concatenates backscatter samples between adjacent bathymetry beam
        bottom detections and interpolates their position. Georeferenced
        samples are appended to the classes numpy array. Bathymetry swathes
        which could not be positioned are excluded.

        Parameters
        ----------
        nextBeam : pyall cBeam
            Next beam to be concatenated.
        """
        # Start swath with bottom detection sample of the first beam
        if self.lastBeam is None:
            # Initialize numpy array
            self.georeferencedSamples = np.array(
                [
                    [
                        self.bathymetryOfSwath.longitude.loc[0],
                        self.bathymetryOfSwath.latitude.loc[0],
                        self.bathymetryOfSwath.depth.loc[0],
                        nextBeam.samples[nextBeam.centreSampleNumber]*0.1
                        ]
                    ]
                )
            # Proceed to next beam
            self.lastBeam = nextBeam
            self.last_beam_index = 0
        else:
            # Accumulate samples between bottom detects of last and next beam
            samples = np.hstack(
                (
                    np.array(
                        self.lastBeam.samples[self.lastBeam.
                                              centreSampleNumber:]
                        )*0.1,
                    np.array(
                        nextBeam.samples[:nextBeam.centreSampleNumber]
                        )*0.1
                    )
                )
            # Get bathymetric information of last and next beam
            longitudes = [
                self.bathymetryOfSwath.longitude.loc[self.last_beam_index],
                self.bathymetryOfSwath.longitude.loc[self.last_beam_index+1]
                ]
            latitudes = [
                self.bathymetryOfSwath.latitude.loc[self.last_beam_index],
                self.bathymetryOfSwath.latitude.loc[self.last_beam_index+1]
                ]
            depths = [
                self.bathymetryOfSwath.depth.loc[self.last_beam_index],
                self.bathymetryOfSwath.depth.loc[self.last_beam_index+1]
                ]
            # Interpolate bathymetry to samples
            x_bathymetry = [0, len(samples)-1]
            x_samples = list(range(len(samples)))
            longitudes = np.interp(x_samples, x_bathymetry, longitudes)
            latitudes = np.interp(x_samples, x_bathymetry, latitudes)
            depths = np.interp(x_samples, x_bathymetry, depths)
            # Append all but the first sample
            self.georeferencedSamples = np.vstack(
                (
                    self.georeferencedSamples,
                    np.array(
                        [
                            longitudes[1:],
                            latitudes[1:],
                            depths[1:],
                            samples[1:]
                            ]
                        ).transpose()
                    )
                )
            # Prepare next beam
            self.lastBeam = nextBeam
            self.last_beam_index += 1
        return self


def decode_raw_backscatter(filename, bathymetry):
    """Decode raw seabed images and return georeferenced backscatter.

    Georeferences the beam time series backscatter (seabed image) data of an
    ALL file with previously created bathymetry.

    Parameters
    ----------
    filename : str
        The file location of the Kongsberg ALL file.
    bathymetry : pandas DataFrame
        Bathymetry of the ALL file.

    Returns
    -------
    backscatter : np array
        Georeferenced backscatter samples.
    """
    backscatter = np.array([[], [], [], []]).transpose()

    data = pyall.ALLReader(filename)
    while data.moreData():
        typeOfDatagram, datagram = data.readDatagram()

        # Read Y seabed image data
        if typeOfDatagram == 'Y':
            datagram.read()
            # Find the swath in the bathymetry via the ping counter
            try:
                bathymetry_swath = bathymetry.loc[datagram.Counter]
            # Ignore the swath when no bathymetry is available
            except KeyError:
                continue

            # Else georeference samples of seabed image datagram
            backscatter = np.vstack(
                (backscatter,
                 reduce(
                     lambda acc, beam: acc.update(beam),
                     datagram.beams,
                     GeoReferencingAccumulator(bathymetry_swath)
                     ).georeferencedSamples
                 )
                )
            continue
    print(f'Georeferenced {len(backscatter)} backscatter samples.')
    return backscatter


def UTM_EPSG(longitude, latitude):
    """Find UTM EPSG code of a longitude, latitude point."""
    zone = str((math.floor((longitude + 180) / 6) % 60) + 1)
    epsg_code = 32600
    epsg_code += int(zone)
    if latitude < 0:
        epsg_code += 100
    return epsg_code


def main(filename, backscatter=True):
    """Run functions."""
    # Decode raw bathymetry
    positions, XYZs = decode_raw_bathymetry(filename)
    # Interpolate navigation to ping time
    navigated_XYZs = process_navigation(positions, XYZs)
    # Transform soundings from vessel to geographic coordinates
    bathymetry = locate_soundings(navigated_XYZs)
    print(f'Dataset includes {len(bathymetry)} soundings.')
    if backscatter:
        # Georeference backscatter
        backscatter = decode_raw_backscatter(filename, bathymetry)
        return bathymetry, backscatter
    else:
        return bathymetry


if __name__ == "__main__":

    bathymetry, backscatter = main('./test/EM122_data.all')

    bathymetry.to_csv(
        './test/bathymetry.csv',
        sep=',',
        columns=[
            'longitude',
            'latitude',
            'depth',
            'classification'
            ],
        header=[
            'X',
            'Y',
            'Z',
            'Classification'
            ],
        index=False,
        mode='w'
        )

    np.savetxt(
        './test/backscatter.csv',
        backscatter,
        delimiter=',',
        header='X,Y,Z,Amplitude',
        comments=''
        )
