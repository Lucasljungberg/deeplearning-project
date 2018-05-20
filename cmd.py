import sys

class CmdArgs:

    valid_models = [
        'vgg16',
        'vgg19',
        'resnet101',
        'resnet152'
    ]

    valid_modes = [
        'train',
        'evaluate'
    ]

    valid_flags = [
        '--plot',
        '--save-train-data'
    ]

    def __init__ (self):
        self.model = None
        self.mode = None
        self.plot = False
        self.save_train_data = False

    def set (self, flag):
        if flag == '--save-train-data':
            self.save_train_data = True
        elif flag == '--plot':
            self.plot = True

def validate_cmdargs (args):
    cmdargs = CmdArgs()
    def print_usage ():
        print('Usage: python main.py <%s> <%s> [--plot] [--save-train-data]' 
            %('|'.join(CmdArgs.valid_modes), '|'.join(CmdArgs.valid_models)))

    # Exit early if no mode is given
    if len(sys.argv) < 3:
        print('A mode and model needs to be specified')
        print_usage()
        exit()

    arg1 = sys.argv[1]
    if arg1 not in CmdArgs.valid_modes:
        print('Unknown mode', arg1)
        print_usage()
        exit()
    cmdargs.mode = arg1

    arg2 = sys.argv[2]
    if arg2 not in CmdArgs.valid_models:
        print('Unknown model', arg2)
        print_usage()
        exit()
    cmdargs.model = arg2

    for arg in sys.argv[3:]:
        if arg not in CmdArgs.valid_flags:
            print('Unknown option', arg)
            print_usage()
            exit()
        cmdargs.set(arg)
    return cmdargs
