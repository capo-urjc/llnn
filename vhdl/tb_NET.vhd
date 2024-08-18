-----------------------------------------------------------------------------------------
                -- Module Name: LUTNN (Testbench) --
-----------------------------------------------------------------------------------------
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use work.Globals.all;

-----------------------------------------------------------------------------------------
-- ENTITY
-----------------------------------------------------------------------------------------
entity tb_NET is
end tb_NET;

-----------------------------------------------------------------------------------------
-- ARCHITECTURE
-----------------------------------------------------------------------------------------
architecture Behavioral of tb_NET is
    -- Component declaration
    component top is
        Port ( NET_I : in  STD_LOGIC_VECTOR(NET_INPUTS-1 downto 0);
               NET_O : out STD_LOGIC_VECTOR(NET_OUTPUT-1 downto 0)
        );
    end component;
    
    -- Internal signals
    signal NET_I : STD_LOGIC_VECTOR(NET_INPUTS-1 downto 0);
    signal NET_O : STD_LOGIC_VECTOR(NET_OUTPUT-1 downto 0);
    
    -- Test inputs (binarized and reversed 20x20 MNIST images)
    constant MNIST_seven : STD_LOGIC_VECTOR(NET_INPUTS-1 downto 0) := x"003C0007C0007C00078000F8000F0001F0001E0003C0003800078000F0000F0001E0001E0003C0003FF803FFFC3FFFC000FC";
    constant MNIST_two   : STD_LOGIC_VECTOR(NET_INPUTS-1 downto 0) := x"03FFCFFFFEFFFFEFFBFE0001E0003E0003E0007C00078000F8001F0001E0003E0007C0007860078F007BF007FE007FC003F8";
    constant MNIST_one   : STD_LOGIC_VECTOR(NET_INPUTS-1 downto 0) := x"003C0003C000780007800070000700007000070000E0000E0001E0001C0001C0001C00038000380003800030000700007000";
    constant MNIST_zero  : STD_LOGIC_VECTOR(NET_INPUTS-1 downto 0) := x"00FE000FF807FFC07FFC07FFC0FC1C1F81C3F81C3E01C3E01C3E07C3E0FC3F3FC0FFF807FF007FE003FE000FC000F8000F80";
    constant MNIST_four  : STD_LOGIC_VECTOR(NET_INPUTS-1 downto 0) := x"03000070000700007000070000F000073E00FFF80FFF80F03C0E01C0E01C1E0383C0383C070380F0380E0380E0180C0180C0";
begin
    -- Component instantiation
    DUT : top port map(NET_I => NET_I, NET_O => NET_O);
    
    -- Process to test the network
    process begin
        NET_I <= MNIST_seven; wait for 20ns;
        NET_I <= MNIST_two;   wait for 20ns;
        NET_I <= MNIST_one;   wait for 20ns;
        NET_I <= MNIST_zero;  wait for 20ns;
        NET_I <= MNIST_four;  wait;
    end process;
end Behavioral;
