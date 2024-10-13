import re
from datetime import datetime
import csv
import json

def get_ordered_syllable_for_song(song_syllable_onsets_offsets_ms):
    """Using syllable_onsets_offsets_ms dictionary return an ordered list of tuples (syllable_label, onset, offset)

    syllable_onsets_offsets_ms (dict)
        key: syllable_label
        value: list of tuples (onset time, offset time)

    Alas python dictionaries are not ordered, so we can't rely on the order of the keys.
    """
    raw_syllable_tuples = []
    for syllable_label, times in song_syllable_onsets_offsets_ms.items():
        for start, end in times:
            raw_syllable_tuples.append((syllable_label, start, end))

    sorted_syllable_tuples = sorted(raw_syllable_tuples, key=lambda x: x[1])
    return sorted_syllable_tuples



def get_recording_time_from_filename(recording_file_path_name):
    """Function to extract animal_id and convert date/time to a datetime object using named groups"""
    try:
        # Define the regex pattern with named groups for animal_id, month, day, hour, minute, and second
        pattern = r'(?P<animal_id>[\w\d]+)_\d+\.\d+_(?P<month>\d+)_(?P<day>\d+)_(?P<hour>\d+)_(?P<minute>\d+)_(?P<second>\d+)\.wav$'

        # Search for the pattern in the file path
        match = re.search(pattern, recording_file_path_name)

        if match:
            # Use the named groups to extract the values
            animal_id = match.group('animal_id')
            month = match.group('month').zfill(2)
            day = match.group('day').zfill(2)
            hour = match.group('hour').zfill(2)
            minute = match.group('minute').zfill(2)
            second = match.group('second').zfill(2)

            # Construct a datetime object (assuming the year is 2024 for this example)
            date_time_str = f"2024-{month}-{day} {hour}:{minute}:{second}"
            date_time_obj = datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S')

            return animal_id, date_time_obj
        else:
            return None, None  # Return None if no match is found
    except Exception as e:
        print(f"Error: {e}")
        return None, None


def load_single_bird_syllable_csv(file_path):
    """"""
    def unescape_and_eval(v):
        if v.startswith("''"):
            v = v.replace("''", "")
        if v.startswith("'"):
            v = v.replace("'", "")

        return eval(v)

    results = []
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)

        for row in reader:
            syllable_onsets_offsets_timebins = unescape_and_eval(row['syllable_onsets_offsets_timebins'])
            syllable_onsets_offsets_ms = unescape_and_eval(row['syllable_onsets_offsets_ms'])

            ordered_and_timed_syllables = get_ordered_syllable_for_song(syllable_onsets_offsets_ms)
            animal_id, recording_time = get_recording_time_from_filename(row['file_name'])

            data = {
                "file_name": row['file_name'],
                "song_present": row['song_present'],
                #'syllable_onsets_offsets_timebins': syllable_onsets_offsets_timebins,
                #'syllable_onsets_offsets_ms': syllable_onsets_offsets_ms,

                'animal_id': animal_id,
                'recording_time': recording_time,

                'ordered_and_timed_syllables': ordered_and_timed_syllables
            }

            results.append(data)

    return results
