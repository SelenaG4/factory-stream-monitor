from stream.schema import EVENT_SCHEMA, JSON_READ_SCHEMA
from stream.config import SENSORS


def test_read_schema_has_all_fields():
    names = [f.name for f in JSON_READ_SCHEMA.fields]
    for base in ["event_id", "event_time", "site", "machine_id", "machine_type"]:
        assert base in names
    for s in SENSORS:
        assert s in names
    # event_time is read as string (parsed with to_timestamp in the job)
    et = [f for f in JSON_READ_SCHEMA.fields if f.name == "event_time"][0]
    assert et.dataType.typeName() == "string"


def test_event_schema_types_the_timestamp():
    et = [f for f in EVENT_SCHEMA.fields if f.name == "event_time"][0]
    assert et.dataType.typeName() == "timestamp"
