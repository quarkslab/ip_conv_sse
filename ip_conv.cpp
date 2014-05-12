/* 
 * Copyright (c) 2014, Adrien Guinet <aguinet@quarkslab.com>
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * - Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * - Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * - Neither the name of Quarkslab nor the names of its contributors may be used
 * to endorse or promote products derived from this software without specific
 * prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#define _BSD_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <sys/time.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include <immintrin.h>

#include <iostream>

char** init_strs(size_t n)
{
	char** strs = new char*[n];
	for (size_t i = 0; i < n; i++) {
		uint32_t ip = rand() * rand();
		const char* ip_s = inet_ntoa(*reinterpret_cast<in_addr*>(&ip));
		char* ip_dup;
		posix_memalign((void**) &ip_dup, 16, 17);
		strncpy(ip_dup, ip_s, 17);
		strs[i] = ip_dup;
	}
	return strs;
}

void free_strs(char** strs, size_t n)
{
	for (size_t i = 0; i < n; i++) {
		free(strs[i]);
	}
	free(strs);
}

uint32_t atoi3(const char* str, const size_t len_int)
{
	static int pow10[] = {1, 10, 100};

	if (len_int == 0 || len_int >= 4) {
		return -1;
	}   
	uint32_t cur_int = 0;
	for (size_t j = 0; j < len_int; j++) {
		const char c = str[j];
		if ((c < '0') || (c > '9')) {
			return -1;
		}
		cur_int += (c-'0')*pow10[len_int-j-1];
	}   
	return cur_int;
}

uint32_t ipv4toi(const char* str, const size_t size, bool& valid, int min_dots)
{
	if (size == 0) {
		valid = false;
		return 0;
	}

	uint32_t ret = 0;
	int cur_idx = 3;
	size_t start_idx = 0;
	for (size_t i = 0; i <= size; i++) {
		const char c = str[i];
		if (c == '.' || (i == size)) {
			if (cur_idx < 0) {
				valid = false;
				return 0;
			}

			const uint32_t cur_int = atoi3(&str[start_idx], i-start_idx);
			if (cur_int > 0xFF) {
				valid = false;
				return 0;
			}
			ret |= cur_int << (8*cur_idx);
			cur_idx--;
			start_idx = i+1;
			continue;
		}

		if ((c < '0') || (c > '9')) {
			valid = false;
			return 0;
		}
	}

	valid = (cur_idx <= (2-min_dots));
	return ret;
}

uint32_t ipv4toi(const char* str, bool& valid, int min_dots = 3)
{
	return ipv4toi(str, strlen(str), valid, min_dots);
}

bool ip_conv_sse(const char* str, uint32_t* res)
{
	// Load the IP string into an SSE register
	__m128i sse_str = _mm_load_si128((__m128i*)str);

	/*
	 * Initialise some constant values
	 */
	
	// Vector full of '.'.
	__m128i _mask_all_pts = _mm_set1_epi8('.');

	// Vector full of '0'
	__m128i _sse_zerochar = _mm_set1_epi8('0');

	// Vectors full of 0,1,2
	__m128i _sse_second = _mm_set1_epi8(2);
	__m128i _sse_first = _mm_set1_epi8(1);
	__m128i _sse_zero = _mm_setzero_si128();

	// Shuffle vectors for left and right shiftings. Used by the "shifting
	// dance" part of the algorithm.
	__m128i _sse_leftshift = _mm_set_epi8(0x80,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1);
	__m128i _sse_rightshift[4];
	_sse_rightshift[0] = _mm_set_epi8(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);
	_sse_rightshift[1] = _mm_set_epi8(14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,0x80);
	_sse_rightshift[2] = _mm_set_epi8(13,12,11,10,9,8,7,6,5,4,3,2,1,0,0x80,0x80);
	_sse_rightshift[3] = _mm_set_epi8(12,11,10,9,8,7,6,5,4,3,2,1,0,0x80,0x80,0x80);
	__m128i sse_rightshift4 = _mm_set_epi8(11,10,9,8,7,6,5,4,3,2,1,0,0x80,0x80,0x80,0x80);

	/*
	 * First step: find out the dots and zeros, and do the initial subtraction
	 */
	
	// mask_dots[i] = (str[i] == '.' or '\0') ? 0xFF:0x00
	__m128i mask_dots = _mm_or_si128(_mm_cmpeq_epi8(sse_str, _mask_all_pts), _mm_cmpeq_epi8(sse_str, _sse_zero));

	// sse_sub = ~mask_dots & {'0','0',...}
	__m128i sse_sub = _mm_andnot_si128(mask_dots, _sse_zerochar);
	__m128i sse_ip_int = _mm_andnot_si128(mask_dots, _mm_sub_epi8(sse_str, sse_sub));
	
	/*
	 * Now, let's dance!
	 */
	
	// Shift by one to the left the mask_dots vector
	__m128i mask_shifts_count = _mm_shuffle_epi8(mask_dots, _sse_leftshift);
	__m128i sse_shifts_count = _mm_and_si128(mask_shifts_count, _sse_second);

	// mask_shifts_count = ~mask_shifts_count & mask_shifts_count << 1.
	mask_shifts_count = _mm_andnot_si128(mask_shifts_count, _mm_shuffle_epi8(mask_shifts_count, _sse_leftshift));

	// sse_shifts_count |= mask_shifts_count & _sse_first
	sse_shifts_count = _mm_or_si128(sse_shifts_count, _mm_and_si128(mask_shifts_count, _sse_first));


	/*
	 * Ok `shifts_count' vector is computed. Let's go on with dance the and
	 * make our shifts and extract the final part of the IP
	 */
	__m128i sse_shift_vec;
	__m128i sse_ip_final;
	__m128i sse_byte_mask = _mm_set_epi32(0,0,0,-1);
	int shift;
	
#define DO_SHIFT\
	sse_shift_vec = _sse_rightshift[shift];\
	sse_ip_int = _mm_shuffle_epi8(sse_ip_int, sse_shift_vec);\
	sse_shifts_count = _mm_shuffle_epi8(sse_shifts_count, sse_shift_vec);\

#define FINALIZE_MOVE\
	sse_ip_int = _mm_andnot_si128(sse_byte_mask, sse_ip_int);\
	sse_byte_mask = _mm_shuffle_epi8(sse_byte_mask, sse_rightshift4);

	shift = _mm_extract_epi8(sse_shifts_count, 0);
	DO_SHIFT
	sse_ip_final = _mm_and_si128(sse_ip_int, sse_byte_mask);
	FINALIZE_MOVE

	shift = _mm_extract_epi8(sse_shifts_count, 4);
	DO_SHIFT
	sse_ip_final = _mm_or_si128(sse_ip_final, _mm_and_si128(sse_ip_int, sse_byte_mask));
	FINALIZE_MOVE

	shift = _mm_extract_epi8(sse_shifts_count, 8);
	DO_SHIFT
	sse_ip_final = _mm_or_si128(sse_ip_final, _mm_and_si128(sse_ip_int, sse_byte_mask));
	FINALIZE_MOVE

	shift = _mm_extract_epi8(sse_shifts_count, 12);
	DO_SHIFT
	sse_ip_final = _mm_or_si128(sse_ip_final, _mm_and_si128(sse_ip_int, sse_byte_mask));
	FINALIZE_MOVE

	/*
	 * Final computations
	 */
	__m128i sse_mul = _mm_set_epi8(0,1,10,100,0,1,10,100,0,1,10,100,0,1,10,100);
	sse_ip_final = _mm_maddubs_epi16(sse_ip_final, sse_mul);

	sse_mul = _mm_set_epi16(1,1,1<<8,1<<8, 1,1,1<<8,1<<8);
	sse_ip_final = _mm_madd_epi16(sse_ip_final, sse_mul);

	uint32_t __attribute__((aligned(16))) ip_res_tmp[4];
	_mm_store_si128((__m128i*)ip_res_tmp, sse_ip_final);
	*res = ((ip_res_tmp[0] | ip_res_tmp[1]) << 16) |
		    (ip_res_tmp[2] | ip_res_tmp[3]);

	return true;
}

template <class Func>
bool verify(const char* name, char** strs, size_t n, Func const& f)
{
	bool ret = true;
	bool valid;
	for (size_t i = 0; i < n; i++) {
		const char* s = strs[i];
		const uint32_t ref = ipv4toi(s, valid);
		const uint32_t v = f(s);
		if (v != ref) {
			std::cerr << std::hex << "Error with " << s << ", expected " << ref << ", got " << v << " instead." << std::endl;
			ret = false;
		}
	}
	return ret;
}

template <class Func>
bool bench(const char* name, char** strs, size_t n, Func const& f)
{
	if (!verify(name, strs, n, f)) {
		return false;
	}
	struct timeval start,end;
	gettimeofday(&start, NULL);
	uint32_t v;
	for (size_t i = 0; i < n; i++) {
		v = f(strs[i]);
	}
	gettimeofday(&end, NULL);
	double diff = (end.tv_sec+(double)end.tv_usec/(double)1000000) - (start.tv_sec+(double)start.tv_usec/(double)1000000);
	std::cout << std::fixed << name << ": " << diff << " s, " << (n/diff) << " conversions/s" << std::endl;

	return true;
}

int main(int argc, char** argv)
{
	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " n" << std::endl;
		return 1;
	}

	srand(time(NULL));

	const size_t n = atoll(argv[1]);
	std::cerr << "Generating random IP strings..."; 
	std::cerr.flush();
	char** strs = init_strs(n);
	std::cerr << " done." << std::endl;

	bench("inet_aton", strs, n,
		[](const char* s)
		{
			uint32_t v;
			inet_aton(s, (in_addr*) &v);
			return ntohl(v);
		});

	bench("leeloo", strs, n,
		[](const char* s)
		{
			bool valid;
			return ipv4toi(s, valid);
		});

	bench("sse", strs, n,
		[](const char* s)
		{
			uint32_t res;
			ip_conv_sse(s, &res);
			return res;
		});

	free_strs(strs, n);

	return 0;
}
