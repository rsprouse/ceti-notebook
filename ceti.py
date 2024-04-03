import sys, re
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from parselmouth import Sound
from parselmouth.praat import call as pcall
from scipy.signal import welch, butter, filtfilt

from audiolabel import df2tg
from phonlab.utils import dir2df, get_timestamp_now
from phonlab.array import nonzero_groups

def hhmmss2sec(s):
    '''
    Convert a HH:MM:SS string to seconds.
    '''
    hh, mm, ss = s.split(':')
    return int(hh) * 3600 + int(mm) * 60 + int(ss)

def butter_highpass(cut, fs, order=3):
    nyq = 0.5 * fs
    cut = cut / nyq
    b, a = butter(order, cut, btype='high')
    return b, a

def butter_highpass_filter(data, cut, fs, order=3):
    b, a = butter_highpass(cut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def praat_ltas(snd, bwhz, return_freqs):
    ltas = pcall(snd, 'To Ltas...', bwhz)
    nbins = pcall(ltas, 'Get number of bins')
    vals = [pcall(ltas, 'Get value in bin', i) for i in range(1, nbins+1)]
    if return_freqs is True:
        freqs = [
            pcall(ltas, 'Get frequency from bin number...', i) for i in range(1, nbins+1)
        ]
        return (vals, freqs)
    else:
        return vals

def load_flac_info(flacdir):
    '''
    Find all .flac files and load as a dataframe with samplerate and durations.
    
    If cached results are available, use those instead.
    '''
    flacinfofile = flacdir / 'flacinfo.csv'
    if flacinfofile.exists():
        dtypes = {
            'relpath': str,
            'fname': str,
            'barename': str,
            'flacrate': int,
            'flacdur': float,
            'tag': str,
            'fseq': int,
            'foffset': float
        }
        flacdf = pd.read_csv(flacinfofile, dtype=dtypes)
    else:
        flacdf = dir2df(flacdir, fnpat='(?P<tag>[a-z]+\d+[a-z])(?P<fseq>\d+)\.flac$', dirpat=r'20\d\d', addcols=['barename', 'ext'])
        flacdf['fseq'] = flacdf['fseq'].astype(int)
        flacdf['flacrate'] = [sf.info(flacdir / f.relpath / f.fname).samplerate for f in flacdf.itertuples()]
        flacdf['flacdur'] = [sf.info(flacdir / f.relpath / f.fname).duration for f in flacdf.itertuples()]
        flacdf['foffset'] = flacdf.groupby('tag', group_keys=False).apply(
            lambda x: x['flacdur'].shift(fill_value=0).cumsum()
        )
        flacdf.to_csv(flacinfofile, index=False)
    return flacdf

def load_classified_codas(url, dropIPI, jocastacorrections, conversation_sec):
    '''
    Load codas found in google classifiedCodasProc spreadsheet as a dataframe.
    '''
    codas = pd.read_csv(
        url, dtype={'codaNUM2018': str, 'IDN': str, 'nClicks': np.int64}
    )
    # Correct bad Jocasta coda values
    jocidx = codas['codaNUM2018'].isin(jocastacorrections['codas'])
    codas.loc[jocidx, 'TsTo'] = codas[jocidx]['TsTo'] - jocastacorrections['offset']

    if dropIPI is True:   # Drop IPI# columns if requested
        codas = codas.drop(
            [c for c in codas.columns if c.startswith('IPI')],
            axis='columns'
        )

    # Add datetime of coda as `codadt` column.
    try:
        assert(codas['Date'].str.endswith(' 00:00:00').all())
    except AssertionError:
        raise('Did not expect non-zero time in the `Date` column')
    
    codas['codadt'] = pd.to_datetime(
        codas['Date'].str.replace('00:00:00', '') + codas['TagOnTime'], # concatenate Date and TagOnTime
        format='%Y-%m-%d %H:%M:%S'
    ) + pd.to_timedelta(codas['TsTo'], unit='seconds') # Add TsTo

    # Label conversations with consecutive integers (-1 for not in conversation).
    # Conversations are series of codas no more than `conversation_sec` seconds
    # apart.
    codas = codas.sort_values('codadt')
    convoidx = nonzero_groups(
        codas['codadt'].diff() < pd.to_timedelta(conversation_sec, unit='seconds')
    )
    convoidx = [[i[0] - 1] + list(i) for i in convoidx] # diff() doesn't capture first coda of the conversation, so add it back
    addidx = np.full(len(codas), -1, dtype=int)
    for i, idx in enumerate(convoidx):
        addidx[idx] = i
    codas['convo'] = addidx

    # Add number of conversation partipants as `convoN`.
    codas['convoN'] = 0
    for g in codas[codas['convo'] >= 0].groupby('convo'):
        codas.loc[g[1].index, 'convoN'] = len(g[1]['Name'].unique())
    
    barenamere = re.compile(r'''
        (?P<barename>             # Corresponds to .flac filename
          (?P<tagmatch>sw\d+[a-z]) # Expected to match 'Tag' column
          (?P<fileseq>\d+)        # Sequence of this file in audio files for Tag
        )
        (?:
          _                     # Separator for optional
          (?P<seg>.+)           # following digits of unknown meaning
        )?
    ''', re.VERBOSE)
    codas = pd.concat(
        [
            codas['REC'].str.extract(barenamere).fillna(''),
            codas
        ],
        axis='columns'
    )
    codas = codas.sort_index()
    assert((codas['tagmatch'] == codas['Tag']).all())
    return codas.drop('tagmatch', axis='columns')

def codadf2clickdf(codadf, codacol, t1col):
    '''
    Convert a wide coda dataframe in which rows represent codas and in which
    individual clicks times are in ICI# columns to a long format in which rows
    are the individual clicks.
    '''
    df = pd.wide_to_long(
        codadf,
        stubnames='ICI',
        i=codacol,   # Name of column with coda id
        j='clicknum' # Name of column in output with the ordered number of the click within each coda
    ) \
    .reset_index() \
    .groupby(codacol).apply(
        lambda x: x[(x['ICI'] != 0) | ~x['ICI'].duplicated()]  # Remove duplicated ICI rows == 0.0
    ) \
    .reset_index(drop=True)
    # Add annotated t1 of the click
    df['t1'] = df.groupby(codacol, group_keys=False).apply(
        lambda x: x['ICI'].shift(fill_value=x.iloc[0][t1col]).cumsum()
    )
    return df

def load_cached_spectra(csvfile):
    '''
    Load spectral analysis from cached .npy and .csv files.
    '''
    return {
        'wfreqs': np.load(csvfile.with_suffix('.welchfreqs.npy')),
        'pfreqs': np.load(csvfile.with_suffix('.praatfreqs.npy')),
        'lfreqs': np.load(csvfile.with_suffix('.ltasfreqs.npy')),
        'au': np.load(csvfile.with_suffix('.audio.npy')),
        'welch': np.load(csvfile.with_suffix('.welchspec.npy')),
        'praat': np.load(csvfile.with_suffix('.praatspec.npy')),
        'ltas': np.load(csvfile.with_suffix('.ltasspec.npy')),
        'md': pd.read_csv(
            csvfile,
            dtype={'codaNUM2018': str, 'clicknum': int, 'text': str, 'IDN': str}
        )
    }

def get_single_click_audio(aufile, t1, click_offset, click_window, resample_rate, verbose=True):
    with sf.SoundFile(aufile, 'r') as fh:
        sr_native = fh.samplerate
        nsamp = int(click_window * sr_native)
        fh.seek(int((t1 + click_offset) * sr_native))
        # Load the target number of frames, and transpose to match librosa form
        y = fh.read(frames=nsamp, dtype=np.float32, always_2d=False).T
        if sr_native != resample_rate:
            y = librosa.resample(y, orig_sr=sr_native, target_sr=resample_rate)
        if verbose is True:
            print(f'Read from time {t1 + click_offset} in {aufile}')
    return y

def extract_click_audio(clickdf, codacol, t1col, click_offset, click_window,
    audiodir, resample_rate, groupby):
    '''
    Extract audio chunks for each click row in a dataframe.
    
    For better IO performance reads are grouped by audio file, indicated by
    the `groupby` param.
    '''
    audio = None
    for _, audf in clickdf.groupby(groupby):
        aufile = Path(audiodir) / \
                 audf.iloc[0]['relpath'] / \
                 f"{audf.iloc[0][groupby]}{audf.iloc[0]['ext']}"
        with sf.SoundFile(aufile, 'r') as fh:
            sr_native = fh.samplerate
            if sr_native < resample_rate:
                sys.stderr.write(
                    f'WARNING: Upsampling {aufile} from {sr_native} to {resample_rate}.\n'
                )
            nsamp = int(click_window * sr_native)
            for __, cdf in audf.groupby(codacol):
                for row in cdf.itertuples():
                    try:
                        fh.seek(int((row.t1 + click_offset) * sr_native))
                        # Load the target number of frames, and transpose to match librosa form
                        y = fh.read(frames=nsamp, dtype=np.float32, always_2d=False).T
                        if sr_native != resample_rate:
                            y = librosa.resample(
                                y, orig_sr=sr_native, target_sr=resample_rate
                            )
                        if audio is None:
                            audio = (
                                np.empty(
                                    (len(clickdf), ) + y.shape,
                                    dtype=np.float32
                                ) * np.nan
                            )
                        audio[row.audidx] = y
                    except Exception as e:
                        pass
    return audio

def normalize_audio(audio, remdc, peak_scale, axis=-1):
    '''
    Normalize audio ndarray along axis (last axis by default).
    '''
    # Remove DC offset.
    if remdc is True:
        mu = audio.mean(axis=axis, keepdims=True)
        audio -= mu
    # Peak normalization.
    if peak_scale is not None:
#        audio /= np.abs(audio).max(axis=axis, keepdims=True) * peak_scale
        audio *= peak_scale / np.abs(audio).max(axis=axis, keepdims=True)
    return audio

# TODO: as-is this is too particular for a specific dataframe
def clicks2tg(mydf):
    '''
    Compile textgrid tiers from dataframe of clicks.
    '''
    clickdf = pd.DataFrame({
        'whale': mydf['Name'],
        'coda': mydf['codaNUM2018'],
        'click': mydf['clicknum'].astype(str),
        't1': mydf.index * click_window,
        't2': (mydf.index + 1) * click_window
    })
    clickdf = clickdf.set_index(['whale', 'coda'])

    whaledf = clickdf[['t1', 't2']] \
              .groupby('whale') \
              .agg([min, max]) \
              .loc[:, [('t1', 'min'), ('t2', 'max')]] \
              .droplevel(1, axis='columns') \
              .reset_index()

    codadf = clickdf[['t1', 't2']] \
              .groupby(['coda']) \
              .agg([min, max]) \
              .loc[:, [('t1', 'min'), ('t2', 'max')]] \
              .droplevel(1, axis='columns') \
              .reset_index()

    clicktg = df2tg(
        [whaledf, codadf, clickdf],
        tnames=['whale', 'coda-bout', 'clicknum'],
        lbl=['whale', 'coda', 'click'],
        fmt='0.4f',
        outfile=specdir.parent / 'allcodas-clicks.focal.20231005.2.fg.TextGrid'
    )

def codas2clicks(codadf):
    '''
    Transform the codas in a dataframe to a list of per-whale
    coda|click dataframes and associated names of the form
    `{whaleid}-(codas|clicks)`.
    '''
    codalists = {}
    for coda in codadf.itertuples():
        codadict = {
            't1': coda.TsTo,
            't2': coda.TsTo + coda.Duration,
        }
        try:
            codalists[f'{coda.IDN}-codas'].append(coda._asdict() | codadict)
        except KeyError:
            codalists[f'{coda.IDN}-codas'] = [coda._asdict() | codadict]
        t1 = coda.TsTo
        for clicknum in np.arange(1, coda.nClicks + 1, dtype=int):
            clickdur = getattr(coda, f'ICI{int(clicknum)}')
            clickdict = {
                't1': t1,
                'clicknum': clicknum,
            }
            try:
                codalists[f'{coda.IDN}-clicks'].append(coda._asdict() | clickdict)
            except KeyError:
                codalists[f'{coda.IDN}-clicks'] = [coda._asdict() | clickdict]
            t1 += clickdur
    dfs = []
    for v in codalists.values():
        vdf = pd.DataFrame(v)
        vdf['nClicks'] = vdf['nClicks'].astype(str)
        if 'clicknum' in vdf.columns:
            vdf['clicknum'] = vdf['clicknum'].astype(str)
        dfs.append(vdf.drop('Index', axis='columns'))
    return (dfs, list(codalists.keys()))

def specs2long(row, welchspecarray, praatspecarray, ltasspecarray, freqs):
    '''
    Arrange spectral measures in long format and combine with click metadata
    keys for later merging.
    '''
    return {
        'codaNUM2018': row.codaNUM2018,
        'clicknum': row.clicknum,
        'binHz': np.hstack((freqs['welch'], freqs['praat'], freqs['ltas'])),
        'binval': np.hstack((
            (10*np.log10(np.abs(welchspecarray[row.Index])/2e-5)),
            (10*np.log10(np.abs(praatspecarray[row.Index,0,:])/2e-5)),
            ltasspecarray[row.Index]
        )),
        'spectype': \
            ['welch'] * len(freqs['welch']) + \
            ['praat'] * len(freqs['praat']) + \
            ['ltas'] * len(freqs['ltas'])
    }

def get_positions_from_tag_times(df, tag, t1, t2, chanstr, resample_rate):
    '''
    Load position data from a tag and return as a time series with a specified sample rate.
    '''
    ## TODO: handle times from subflac files
    tagdf = df[df['tag'] == tag]
    chans = ['dprh'.find(c) for c in chanstr]  # chanstr has mix of characters in 'dprh'
    try:
        assert(len(tagdf) <= 1)
    except AssertionError:
        msg = f'Found multiple .wav files for tag {tag}. Cannot choose one.\n'
        raise RuntimeError(msg)
    try:
        with sf.SoundFile(tagdf.iloc[0]['posabspath'], 'r') as fh:
            sr_native = fh.samplerate
            fh.seek(np.round(t1 * sr_native).astype(int))
            # Load the target number of frames, transpose to match librosa form, and
            # select channels
            nsamp = np.round((t2 - t1) * sr_native).astype(int)
            y = fh.read(frames=nsamp, dtype=np.float32, always_2d=False).T[chans]
            # Resample if necessary
            if sr_native != resample_rate:
                y = librosa.resample(
                    y, orig_sr=sr_native, target_sr=resample_rate
                )
    except Exception as e:
        msg = f'Could not get position data for tag {tag}.\n\n{e}\n\n'
        raise RuntimeError(msg)
    return y