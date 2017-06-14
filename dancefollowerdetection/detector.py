import pickle
import pandas as pd
import numpy as np


def to_frame_id(frame_id):
    """
    Takes a bb_tracker track id and returns the frame id within.
    E.g. to_frame_id('f14083813666064354331d24c1') = 14083813666064354331
    :param frame_id: The track identifier
    :return: The frame id
    """
    return int(frame_id.split('d')[0][1:])


def tracks_to_dataframe(tracks):
    detections = [d for t in tracks for d in t.meta['detections']]
    ids = [t.id for t in tracks for d in t.meta['detections']]  # require same length as detections
    track_lengths = [len(t.ids) for t in tracks for d in t.meta['detections']]  # require same length as detections

    columns = detections[0]._fields
    df = pd.DataFrame(np.array(detections), columns=columns, index=ids)
    df['x'] = df.x.astype(float)
    df['y'] = df.y.astype(float)
    df['timestamp'] = df.timestamp.astype(float)
    df['orientation'] = df.orientation.astype(float)
    df['track_lengths'] = track_lengths
    df.index.rename('track_id', inplace=True)
    df.loc[:, 'cam_id'] = df.meta.apply(lambda x: x['camId'])
    df.drop('meta', axis=1, inplace=True)
    return df


class DanceFollowerDetector:
    """
        Classifies the behavior of a given track as dance, follow or other.
    """

    # vds constants
    window_size = 5  # in seconds

    # stc constants
    threshhold_dance = 0.136425411793117
    threshhold_follow = 267.420834468306
    vds_window_size = 30
    with open('dancefollowerdetection/lr.pkl', 'rb') as f:
        lr = pickle.load(f)

    @staticmethod
    def change_mid_window(n):
        def f(normed_orientations_in_single_track):
            rmean = normed_orientations_in_single_track.rolling(n).mean()
            dfx = pd.concat([rmean, rmean.shift(-n)], axis=1)
            dfx.columns = ['m1', 'm2']
            dfx.loc[:, 'score'] = dfx.product(axis=1, skipna=False) * (- dfx.abs().min(axis=1) / dfx.abs().max(axis=1))
            return dfx

        return f

    @staticmethod
    def angle_diff(d):
        return np.arctan2(np.sin(d), np.cos(d))

    @staticmethod
    def rotate_vectors(x, y, rotation):
        sined = np.sin(rotation)
        cosined = np.cos(rotation)
        normed_x = x * cosined - y * sined
        normed_y = x * sined + y * cosined
        return pd.DataFrame({'forwards': normed_x, 'sidewards': normed_y})

    def make_vds_features(self, track):
        """
        Takes a track and returns the velocity-direction-switch values to detect single iterations.
        Args:
            track: A track with the columns: x, y, orientation and timestamp.
        Returns:
            The velocity-direction-switch values.
        """
        diffed = track[['x', 'y', 'orientation', 'timestamp']].diff()
        diffed.loc[:, 'orientation'] = diffed.orientation.apply(self.angle_diff)
        diffed.loc[:, 'normed_orientation'] = diffed.orientation / diffed.timestamp / 3

        rot_mid = self.change_mid_window(self.window_size * 3 // 2)(diffed.normed_orientation)
        diffed.loc[:, 'vds_turn'] = rot_mid.score

        egocentric = self.rotate_vectors(diffed.y, diffed.x, -track.orientation)
        sidewards_mid = self.change_mid_window(self.window_size * 3 // 2)(egocentric.sidewards)
        diffed.loc[:, 'vds_side'] = sidewards_mid.score

        return diffed[['vds_turn', 'vds_side']].copy()

    def make_stc_features(self, vds):
        """
        Takes the velocity-direction-switch (vds) values for single dance iteration detection and finds multiple
        with a sliding_threshold_count (stc).
        Args:
            vds: The velocity-direction-switch values from `make_vds_features`
        Returns:
            The sliding threshold count values.
        """
        vds.vds_turn = vds.vds_turn > self.threshhold_dance
        vds.vds_side = vds.vds_side > self.threshhold_follow
        if vds.shape[0] < self.vds_window_size:  # sliding window unnecessary, vds is already short
            vds.vds_turn = vds.vds_turn.sum()
            vds.vds_side = vds.vds_side.sum()
            return vds.rename(columns={'vds_turn': 'stc_turn', 'vds_side': 'stc_side'})

        df = vds.rolling(self.vds_window_size).sum().rename(columns={'vds_turn': 'stc_turn', 'vds_side': 'stc_side'})
        df.loc[:, 'stc_sum'] = df.stc_turn + df.stc_side
        return df

    def predict(self, track, boundaries=False):
        """
        Takes a single track as a pandas `DataFrame` and predicts if the track is a dancer, follower or nothing of both.
          The dataframe requires following columns: ['x', 'y', 'orientation', 'timestamp'].
        Args:
            track (pandas.DataFrame):
        Returns: The behavioral prediction.
        """
        track.reset_index(inplace=True, drop=True)
        vds = self.make_vds_features(track)
        stc = self.make_stc_features(vds).dropna(subset=['stc_turn', 'stc_side'])
        stc_max = stc.sort_values('stc_sum', ascending=False).iloc[:1]
        y_pred = self.lr.predict(stc_max[['stc_turn', 'stc_side']])[0]
        index = stc_max.index.values[0]
        if boundaries:
            y = pd.Series(self.lr.predict(stc[['stc_turn', 'stc_side']]), index=stc.index)
            mask = y != 'o'
            # get index for first occurence on left side and right side 'o'
            left_mask = mask.loc[index::-1].cumprod()
            right_mask = mask.loc[index:].cumprod()
            left = left_mask.argmin() + 1 if not left_mask.all() else 0
            right = right_mask.argmin() if not right_mask.all() else right_mask.index.max()
            return left, index, right, y_pred

        return index, y_pred

    def predict_proba(self, track):
        """
        Takes a single track as a pandas `DataFrame` and predicts as probabilities if the track is a dancer, follower or nothing of both.
          The dataframe requires following columns: ['x', 'y', 'orientation', 'timestamp'].
        Args:
            track (pandas.DataFrame):
        Returns: Probabilities of a the behavioral classes
        """
        vds = self.make_vds_features(track)
        stc = self.make_stc_features(vds)
        stc_max = stc.sort_values('stc_sum', ascending=False).iloc[:1]
        y_pred = self.lr.predict_proba(stc_max[['stc_turn', 'stc_side']])[0]
        return stc_max.index.values[0], y_pred