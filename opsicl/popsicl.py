from omuse.community.pop.interface import POP


class popsicl:
    """An omuse-POP SImulation CLass (POPSICL)"""

    def __init__(
        self,
        popsicl=None,
        nworkers: int = None,
        mode: str = None,
        nml_file: str = None,
    ):
        if popsicl is None:
            self.init_popsicl(nworkers, mode, nml_file)
        else:
            self.copy_init_popsicl(popsicl)

    def init_popsicl(self, nworkers: int, mode: str, nml_file: str):
        self._mode = mode
        self._nworkers = nworkers
        self._p = POP(
            number_of_workers=nworkers,
            mode=mode,
            namelist_file=nml_file,
            redirection="none",
        )

    def copy_init_popsicl(self, popsicl):
        self._mode = popsicl._mode
        self._nworkers = popsicl._nworkers
        self._p = popsicl._p
