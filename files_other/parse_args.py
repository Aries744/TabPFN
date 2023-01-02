# def _parse_args(config_parser, parser):
#     # Do we have a config file to parse?
#     args_config, remaining = config_parser.parse_known_args()
#     if args_config.config:
#         with open(args_config.config, 'r') as f:
#             cfg = yaml.safe_load(f)
#             parser.set_defaults(**cfg)

#     # The main arg parser parses the rest of the args, the usual
#     # defaults will have been overridden if config file specified.
#     args = parser.parse_args(remaining)

#     # Cache the args as a text string to save them in the output dir later
#     args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
#     return args, args_text


# import argparse
# import yaml
# config_parser = argparse.ArgumentParser(description='Only used as a first parser for the config file path.')
# config_parser.add_argument('--config')
# parser = argparse.ArgumentParser()
# parser.add_argument('--encoder', default='linear', type=str, help='Specify depending on the prior.')
# parser.add_argument('--y_encoder', default='linear', type=str, 
#                     help='Specify depending on the prior. You should specify this if you do not fuse x and y.')

# Import the library
import argparse
parser = argparse.ArgumentParser()
# Add an argument
parser.add_argument('--encoder', default='linear', type=str, help='Specify depending on the prior.')
parser.add_argument('--y_encoder', default='linear', type=str, help='Specify depending on the prior. You should specify this if you do not fuse x and y.')
# Parse the argument
args = parser.parse_args()
# Print "Hello" + the user input argument
print('encoder,', args.encoder)
print('y_encoder,', args.y_encoder)

# cd my_files/
# python parse_args.py --encoder 'linear' / 'mlp' / 'positional'