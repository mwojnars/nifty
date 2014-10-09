'''
A mock-up that references to nifty.parsing.waxeye copy of the Waxeye runtime and thus makes "import waxeye" possible without global Waxeye installation.
'''

# nifty; whenever possible, use relative imports to allow embedding of the library inside higher-level packages
try: 
    from ..parsing.waxeye import *
except:
    from nifty.parsing.waxeye import *
