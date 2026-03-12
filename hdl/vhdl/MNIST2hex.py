def bin_to_hex(binary_string):
    """
    Convert a binary string to its hexadecimal representation.
    
    :param binary_string: A string representing a binary number.
    :return: A string representing the hexadecimal equivalent of the binary number.
    """
    # Ensure the number of bits is a multiple of 4
    binary_string = binary_string.zfill((len(binary_string) + 3) // 4 * 4)
    
    # Convert binary to hexadecimal
    hex_string = hex(int(binary_string, 2))[2:]
    
    # Ensure the hexadecimal has the correct length
    hex_string = hex_string.zfill(len(binary_string) // 4)
    
    return hex_string.upper()

# Example usage:
if __name__ == "__main__":
    binary_number = "0000001100000001100000000011000000011000000001110000000111000000011100000001110000001111000000011100000011100000001111000001110000000011110000011100000001111000001110000000011100000011100000000111000000111100000011110000000111111111111100000001111111111111000000000111110011100000000000000000111100000000000000001110000000000000000011100000000000000000111000000000000000001110000000000000000011000000"

    # Reverse the binary number, keeping leading zeros
    reversed_binary_number = binary_number[::-1]

    # Convert the reversed binary number to hexadecimal
    hexadecimal_number = bin_to_hex(reversed_binary_number)

    # Print the hexadecimal number
    print(hexadecimal_number)
