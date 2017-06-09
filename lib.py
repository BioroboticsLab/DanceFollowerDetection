import pickle
import pandas as pd
import numpy as np


def to_frame_id(id):
    """
    Takes a bb_tracker track id and returns the frame id within.
    E.g. to_frame_id('f14083813666064354331d24c1') = 14083813666064354331
    :param id: The track identifier
    :return: The frame id
    """
    return int(id.split('d')[0][1:])

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
        Classsifies the behavior of a given track as dance, follow or other.
    """

    # vds constants
    window_size = 5  # in seconds

    # stc constants
    threshhold_dance = 0.136425411793117
    threshhold_follow = 267.420834468306
    vds_window_size = 30
    lr_pickle_string = b'\x80\x03csklearn.linear_model.logistic\nLogisticRegression\nq\x00)\x81q\x01}q\x02(X\x07\x00\x00\x00penaltyq\x03X\x02\x00\x00\x00l1q\x04X\x04\x00\x00\x00dualq\x05\x89X\x03\x00\x00\x00tolq\x06G?\x1a6\xe2\xeb\x1cC-X\x01\x00\x00\x00Cq\x07G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00fit_interceptq\x08\x88X\x11\x00\x00\x00intercept_scalingq\tK\x01X\x0c\x00\x00\x00class_weightq\nNX\x0c\x00\x00\x00random_stateq\x0bNX\x06\x00\x00\x00solverq\x0cX\t\x00\x00\x00liblinearq\rX\x08\x00\x00\x00max_iterq\x0eKdX\x0b\x00\x00\x00multi_classq\x0fX\x03\x00\x00\x00ovrq\x10X\x07\x00\x00\x00verboseq\x11K\x00X\n\x00\x00\x00warm_startq\x12\x89X\x06\x00\x00\x00n_jobsq\x13K\x01X\x08\x00\x00\x00classes_q\x14cnumpy.core.multiarray\n_reconstruct\nq\x15cnumpy\nndarray\nq\x16K\x00\x85q\x17C\x01bq\x18\x87q\x19Rq\x1a(K\x01K\x03\x85q\x1bcnumpy\ndtype\nq\x1cX\x02\x00\x00\x00O8q\x1dK\x00K\x01\x87q\x1eRq\x1f(K\x03X\x01\x00\x00\x00|q NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK?tq!b\x89]q"(X\x01\x00\x00\x00dq#X\x01\x00\x00\x00fq$X\x01\x00\x00\x00oq%etq&bX\x05\x00\x00\x00coef_q\'h\x15h\x16K\x00\x85q(h\x18\x87q)Rq*(K\x01K\x03K\x02\x86q+h\x1cX\x02\x00\x00\x00f8q,K\x00K\x01\x87q-Rq.(K\x03X\x01\x00\x00\x00<q/NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00tq0b\x88C0p\xa0w\xc6\x91F\xea?G\x12\xa7\xa4q\xa3\xd7?\xdb\xe9\xdbd^\x08\xee\xbf\x96\xbf\xdf\xef\xcf\r\xe4\xbf\xcf-qr?\xa2\xea?\xd6\x85\x15\xe9\xbb\xef\xe8\xbfq1tq2bX\n\x00\x00\x00intercept_q3h\x15h\x16K\x00\x85q4h\x18\x87q5Rq6(K\x01K\x03\x85q7h.\x89C\x18\x1dY\xd1,x\x7f\x1b\xc0\xc2\xb7\xc9\xbd\xfaF\x1a\xc0\xb2\xc7\xbc\xbf\xb7\xbf\x1b@q8tq9bX\x07\x00\x00\x00n_iter_q:h\x15h\x16K\x00\x85q;h\x18\x87q<Rq=(K\x01K\x01\x85q>h\x1cX\x02\x00\x00\x00i4q?K\x00K\x01\x87q@RqA(K\x03h/NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00tqBb\x89C\x04\x14\x00\x00\x00qCtqDbX\x10\x00\x00\x00_sklearn_versionqEX\x04\x00\x00\x000.18qFub.'
    lr = pickle.loads(lr_pickle_string)

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

    @staticmethod
    def get_features(window, window_diff):
        features = {}

        gaps = (window_diff.timestamp // 0.36)
        features['gap_time'] = gaps.sum()
        features['gap_occ'] = (gaps > 0).sum()

        features['vds_turn'] = window_diff.vds_turn.iat[0]
        features['vds_side'] = window_diff.vds_side.iat[0]

        return features

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

        ts_cumsum = diffed.timestamp.cumsum().fillna(0).reset_index(drop=True)

        l = []
        left = 0
        while left < track.shape[0]:
            right = np.argmax((ts_cumsum - ts_cumsum.iat[left]) >= self.window_size)
            if right == 0:
                break
            window = track.iloc[left:right]
            window_diff = diffed.iloc[left:right]
            features = self.get_features(window, window_diff)
            l.append(features)
            left += 1

        df_features = pd.DataFrame(l)
        return df_features

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
        df.gap_time = vds.gap_time
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

