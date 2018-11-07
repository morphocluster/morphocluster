"""
API schemas
"""
from marshmallow import Schema, fields

#from morphocluster.extensions import marshmallow as ma


class JobSchema(Schema):
    uuid = fields.Str(dump_only=True)
    created_at = fields.DateTime(dump_only=True)
    status = fields.Str(dump_only=True)
    name = fields.Str(required=True)
    parameters = fields.Raw()
