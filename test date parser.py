"""
from dateutil import parser
import datetime


def test_date_parsing():
    date_string = "2022:08:04 19:12:40"
    print(f"Parsing date string: {date_string}")

    # Try parsing with dateutil.parser
    try:
        parsed_date = parser.parse(date_string)
        print(f"Parsed with dateutil.parser: {parsed_date}")
    except ValueError as e:
        print(f"dateutil.parser failed: {e}")

    # Try parsing with datetime.strptime
    try:
        parsed_date = datetime.datetime.strptime(date_string, "%Y:%m:%d %H:%M:%S")
        print(f"Parsed with datetime.strptime: {parsed_date}")
    except ValueError as e:
        print(f"datetime.strptime failed: {e}")

    # Print dateutil version
    print(f"dateutil version: {parser.__version__}")


if __name__ == "__main__":
    test_date_parsing()
"""

from dateutil import parser
from dateutil import tz
from datetime import datetime


def parse_datetime(date_string, default_timezone=tz.tzlocal()):
    """
    Parse a date/time string into a timezone-aware datetime object.

    :param date_string: The date/time string to parse
    :param default_timezone: The timezone to use if not specified in the string (default: local timezone)
    :return: A timezone-aware datetime object
    """
    try:
        # Parse the date/time string
        dt = parser.parse(date_string)

        # If the parsed datetime is naive (no timezone info), assume the default timezone
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=default_timezone)

        return dt
    except ValueError as e:
        print(f"Error parsing date/time string: {e}")
        return None


# Example usage
examples = [
    "2023-04-01 14:30:00",
    "April 1, 2023 2:30 PM",
    "01/04/23 14:30",
    "2023-04-01T14:30:00+02:00",
    "Saturday, 01-Apr-23 14:30:00 UTC",
    "Sat Apr 1 14:30:00 2023",
    "1680355800",  # Unix timestamp
]

for example in examples:
    result = parse_datetime(example)
    if result:
        print(f"Input: {example}")
        print(f"Parsed: {result}")
        print(f"UTC: {result.astimezone(tz.UTC)}")
        print()