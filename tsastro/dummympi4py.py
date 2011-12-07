#!/usr/bin/env python2.6
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#from __future__ import unicode_literals


class _COMM_WORLD(object):

    def Barrier(self):
        pass

    def Bcast(self, buf, root=0):
        pass

    def Get_rank(self):
        return 0

    def Get_size(self):
        return 1

    def Allgather(self, sendbuf, recvbuf):
        recvbuf[0][:] = sendbuf[0]

    def Gather(self, sendbuf, recvbuf, root=0):
        recvbuf[0][0] = sendbuf[0]

    def Gatherv(self, sendbuf, recvbuf, root=0):
        recvbuf[0][:] = sendbuf[0]


class MPI(object):

    INT = int
    FLOAT = float
    COMM_WORLD = _COMM_WORLD()
