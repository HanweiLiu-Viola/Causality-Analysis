from pathlib import Path
from typing import Union, List, Tuple
import mne
from mne.datasets import sample

import  os
os.environ["MNE_DATA"] = "/home/viola/Docker/Causailty Analysis/Causality-Analysis/mne_data"

def load_raw() -> Tuple[mne.io.Raw, List[int]]:
    """
    Load raw MNE sample dataset and corresponding events.

    Returns:
        raw (mne.io.Raw): Raw signal data.
        events (ndarray): Events array.
    """
    data_path = Path(sample.data_path())
    fname_raw = data_path / "MEG/sample/sample_audvis_filt-0-40_raw.fif"
    fname_event = data_path / "MEG/sample/sample_audvis_filt-0-40_raw-eve.fif"

    raw = mne.io.read_raw_fif(fname_raw, preload=True)
    events = mne.read_events(fname_event)

    return raw, events

def load_eeg(
    raw: mne.io.Raw,
    events,
    exclude: Union[str, List[str]] = "bads",
    set_ref: bool = True,
    resample_rate: int = 100,
    do_epoch: bool = False,
    tmin: float = -0.2,
    tmax: float = 0.5,
    baseline: Union[Tuple[float, float], None] = (None, 0),
    event_id: dict = None
) -> Union[mne.io.Raw, Tuple[mne.io.Raw, mne.Epochs]]:
    """
    Preprocess EEG data from raw.

    Returns:
        raw or (raw, epochs)
    """
    # Pick EEG channels
    picks = mne.pick_types(raw.info, meg=False, eeg=True, exclude=exclude)
    raw.pick(picks)

    # EEG reference
    if set_ref:
        raw.set_eeg_reference('average', projection=False)

    # Resampling
    if resample_rate:
        raw.resample(resample_rate)

    # Epoching
    if do_epoch:
        if event_id is None:
            event_id = {
                "Auditory/Left": 1,
                "Auditory/Right": 2,
                "Visual/Left": 3,
                "Visual/Right": 4,
                "Smiley": 5,
                "Button": 32,
            }

        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            preload=True,
        )
        return raw, epochs

    return raw

def load_meg(
    raw: mne.io.Raw,
    events,
    exclude: Union[str, List[str]] = "bads",
    resample_rate: int = 100,
    do_epoch: bool = False,
    tmin: float = -0.2,
    tmax: float = 0.5,
    baseline: Union[Tuple[float, float], None] = (None, 0),
    event_id: dict = None
) -> Union[mne.io.Raw, Tuple[mne.io.Raw, mne.Epochs]]:
    """
    Preprocess MEG data from raw.

    Returns:
        raw or (raw, epochs)
    """
    # Pick MEG channels
    picks = mne.pick_types(raw.info, meg=True, eeg=False, exclude=exclude)
    raw.pick(picks)

    # Resampling
    if resample_rate:
        raw.resample(resample_rate)

    # Epoching
    if do_epoch:
        if event_id is None:
            event_id = {
                "Auditory/Left": 1,
                "Auditory/Right": 2,
                "Visual/Left": 3,
                "Visual/Right": 4,
                "Smiley": 5,
                "Button": 32,
            }

        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            preload=True,
        )
        return raw, epochs

    return raw

# # Step 1: Load Raw Data
raw, events = load_raw()
#
# # Step 2.1:
# raw_eeg, eeg_epochs = load_eeg(raw.copy(), events, do_epoch=True)
#
# # Step 2.2:
# raw_meg = load_meg(raw.copy(), events, do_epoch=False)

