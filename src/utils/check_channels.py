import os
import mne

def get_channel_names(subject_path):
    """Reads the MEG file for a subject and returns the set of MEG channel names."""
    wm = os.listdir(subject_path)[-1]
    data_path = os.path.join(subject_path, wm)
    raw = mne.io.read_raw_ctf(data_path, preload=False, verbose=False)
    raw.pick_types(meg=True, stim=True, eeg=False, ref_meg=True, verbose=False)
    return set(raw.info['ch_names'])

def check_channel_consistency(root_dir, subject_ids):
    reference_channels = None
    for sid in subject_ids:
        s = f"S{int(sid):02d}"
        nights = os.listdir(os.path.join(root_dir, s))
        for night in nights:
            subject_path = os.path.join(root_dir, s, night)
            if not os.path.isdir(subject_path):
                continue
            try:
                ch_names = get_channel_names(subject_path)
                if reference_channels is None:
                    reference_channels = ch_names
                    print(f"[Reference] Subject {s}, Night {night}, Channels: {len(reference_channels)}")
                else:
                    missing = reference_channels - ch_names
                    extra = ch_names - reference_channels
                    if missing or extra:
                        print(f"[Mismatch] Subject {s}, Night {night}")
                        if missing:
                            print("  Missing Channels:", missing)
                        if extra:
                            print("  Extra Channels:", extra)
                    else:
                        print(f"[OK] Subject {s}, Night {night}")
            except Exception as e:
                print(f"Error loading data for Subject {s}, Night {night}: {e}")

# === Usage ===
root_dir = r'\\memo-15\DataE\MEGA\raw'  # Update this with the path to your data
subject_ids = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]         # Or any specific subject numbers you want to check
check_channel_consistency(root_dir, subject_ids)
