from typing import Dict
from collections import UserDict

ordinalities = ['continuous', 'discrete']


class IoMetaDataEntry(object):
    def __init__(self, pos: int, name: str=None, ordinality: str='continuous'):
        self._pos = pos
        self._name = name

        self.ordinality = ordinality

    @property
    def pos(self) -> int:
        return self._pos

    @property
    def name(self) -> str:
        if self._name is None:
            return f'x{self._pos}'
        else:
            return self._name

    @name.setter
    def name(self, value: str):
        self._name = value

    @property
    def ordinality(self) -> str:
        return self._ordinality

    @ordinality.setter
    def ordinality(self, value: str):
        if value in ordinalities:
            self._ordinality = value
        else:
            raise ValueError(f'{value} is not one of {ordinalities}')


class IoMetaData(object):
    def __init__(self, n_args: int, args: Dict[str, IoMetaDataEntry], kwargs: Dict[str, IoMetaDataEntry]):
        assert len(args) == n_args

        self._args = args
        self._arg_pos = len(args)
        self._n_args = n_args
        self._validate(args)

        self._kwargs = kwargs
        self._n_kwargs = len(kwargs)

    def _next_arg_pos(self):
        self._arg_pos += 1

        return self._arg_pos

    def _next_kwarg_pos(self):
        self._n_kwargs += 1

        return self._n_kwargs

    def add_arg(self, name: str=None, ordinality: str='continuous'):
        pos = self._next_arg_pos()
        entry = IoMetaDataEntry(pos, name, ordinality)
        self._args[entry.name] = entry

    def add_kwarg(self, name: str, ordinality: str='continuous'):
        pos = self._next_kwarg_pos()
        self._kwargs[name] = IoMetaDataEntry(pos, name, ordinality)

    @staticmethod
    def _validate(args: dict):
        for index, (name, arg) in enumerate(args.items()):
            assert arg.pos == index
            assert arg.name == name

    @property
    def n_args(self):
        return self.n_args

    @property
    def n_kwargs(self):
        return self.n_kwargs

    @classmethod
    def default(cls, n_args: int, kwargs: Dict[str, IoMetaDataEntry]):
        new = cls(n_args, {}, kwargs)

        if n_args == 1:
            new.add_arg('x')
        elif n_args == 2:
            new.add_arg('x')
            new.add_arg('y')
        else:
            for idx in range(n_args):
                new.add_arg(f'x{idx}')

        return new
