# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: msg.proto
# Protobuf Python Version: 5.26.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\tmsg.proto\"\x19\n\nMsgRequest\x12\x0b\n\x03msg\x18\x01 \x01(\t\"\x1a\n\x0bMsgResponse\x12\x0b\n\x03msg\x18\x01 \x01(\t23\n\nMsgService\x12%\n\x06GetMsg\x12\x0b.MsgRequest\x1a\x0c.MsgResponse\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'msg_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_MSGREQUEST']._serialized_start=13
  _globals['_MSGREQUEST']._serialized_end=38
  _globals['_MSGRESPONSE']._serialized_start=40
  _globals['_MSGRESPONSE']._serialized_end=66
  _globals['_MSGSERVICE']._serialized_start=68
  _globals['_MSGSERVICE']._serialized_end=119
# @@protoc_insertion_point(module_scope)
