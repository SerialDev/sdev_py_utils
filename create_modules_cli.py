import argparse
import ast
from modular_templating.modules import *


def try_catch(funcall):
    try:
        ast.literal_eval(funcall)
    except Exception as e:
        print("{} : failed to execute".format(funcall))
    return 1


parser = argparse.ArgumentParser(description="Create a new installable module")
parser.add_argument("--module_name", nargs="*", help="Name of the  module")
parser.add_argument("--author", nargs="*", help="Author of the module")
parser.add_argument("--email", nargs="*", help="Email of the author of the  module")
parser.add_argument("--description", nargs="*", help="Description of the module")
parser.add_argument("--version", nargs="*", help="<Optional> version of the module")
parser.add_argument("--url", nargs="*", help="<Optional> URL of the module")
parser.add_argument("--license", nargs="*", help="<Optional> license of the module")

args = parser.parse_args()
module_name = args.module_name[0]
author = " ".join(args.author)
email = args.email[0]
description = " ".join(args.description)
try_catch("version = args.ids[4]")
try_catch("url = args.ids[4]")
try_catch("license = args.ids[5]")

print(module_name, author, email, description)
generate_folder_structure(module_name)
generate_setup(module_name, author, email, description)
generate_importables(module_name)
