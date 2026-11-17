"""Host command shapes, shared by the HTTP endpoint and the data channel."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

class _Cmd(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Say(_Cmd):
    type: Literal["say"]
    text: str


class AddContext(_Cmd):
    type: Literal["add_context"]
    text: str


class End(_Cmd):
    type: Literal["end"]
    graceful: bool = True


class MuteMic(_Cmd):
    type: Literal["mute_mic"]


class UnmuteMic(_Cmd):
    type: Literal["unmute_mic"]


Command = Annotated[Say | AddContext | End | MuteMic | UnmuteMic, Field(discriminator="type")]
command_adapter = TypeAdapter(Command)
