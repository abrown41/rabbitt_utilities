def main():
    from src.rabbitt_utilities import PADamp, read_command_line

    args = read_command_line()
    PADamp(args)


if __name__ == '__main__':
    main()
