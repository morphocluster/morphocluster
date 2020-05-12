"""
API schemas
"""
from marshmallow import Schema, fields

# from morphocluster.extensions import marshmallow as ma


class RQJobSchema(Schema):
    id = fields.Str(dump_only=True)
    enqueued_at = fields.DateTime(dump_only=True)
    status = fields.Str(dump_only=True)
    func_name = fields.Str(required=True)
    args = fields.List(fields.Raw)
    kwargs = fields.Dict(keys=fields.Str, values=fields.Raw, default=dict)
    result = fields.Raw(dump_only=True)
    exc_info = fields.Raw(dump_only=True)
    description = fields.Raw(dump_only=True)


class JobSchema(Schema):
    name = fields.Str(required=True)

    # Creation only
    args = fields.List(fields.Raw, load_only=True)
    kwargs = fields.Dict(
        keys=fields.Str, values=fields.Raw, default=dict, load_only=True
    )

    # Dump only
    id = fields.Str(dump_only=True)
    job = fields.Nested(RQJobSchema, dump_only=True)
    description = fields.Str(default="", dump_only=True)


class LogSchema(Schema):
    action = fields.Str(required=True)

    node_id = fields.Int(missing=None)
    reverse_action = fields.Str(missing=None)
    data = fields.Raw(missing=None)
