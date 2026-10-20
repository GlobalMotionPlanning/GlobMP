"""dex_deploy — standalone deployment package for the Dexmate CROM3BENGINE online policy.

Import-side-effect free: importing this package must NOT pull in torch, crom_3b_engine, or
any robot package. Heavy symbols live in submodules (e.g. ``deploy_interface``)
and are imported explicitly by callers.
"""

__version__ = "0.1.0"
