#include "Utils.hpp"

int reverse_int32 (int i){
	unsigned char byte1, byte2, byte3, byte4;
	byte1 = i&MAXBYTE;
	byte2 = (i>>8)&MAXBYTE;
	byte3 = (i>>16)&MAXBYTE;
	byte4 = (i>>24)&MAXBYTE;
	return ( (int)byte1<<24 ) + ( (int)byte2<<16 ) + ( (int)byte3<<8 ) + (int)byte4;
}