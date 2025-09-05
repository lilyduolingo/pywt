import sys

from pywt import cli


def main(exec_name: str, command: str, *args: str) -> None:
    parse_args = cli._parse_args(*args)
    match command:
        case "view":
            return cli.View.new(*parse_args.args)(**parse_args.kwargs)


if __name__ == '__main__':
    main(*sys.argv)
