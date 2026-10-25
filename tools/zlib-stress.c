// SPDX-License-Identifier: GPL-2.0-only WITH LicenseRef-Kaiba-Runtime-Exception
// Deterministic deflate/inflate stress driver for tools/zlib-check.sh (not part of the compiler).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "zlib.h"
static unsigned long long rng = 88172645463325252ULL;
static unsigned next(void){ rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (unsigned)(rng >> 16); }
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + t.tv_nsec*1e-9; }
static void gen(unsigned char *p, size_t n, int kind){
  if(kind==0){ const char *w[]={"the ","quick ","brown ","fox ","jumps ","over ","lazy ","dog ","compiler ","kaiba ","llcr ","\n"}; size_t i=0; while(i<n){ const char*s=w[next()%12]; size_t l=strlen(s); for(size_t k=0;k<l&&i<n;k++) p[i++]=s[k]; } }
  else if(kind==1){ for(size_t i=0;i<n;i++) p[i]=(unsigned char)next(); }
  else if(kind==2){ memset(p,0,n); }
  else { for(size_t i=0;i<n;i++) p[i]=(unsigned char)((i*7 + (i>>5)) ^ ((i&1023)==0?next():0)); }
}
int main(int argc, char **argv){
  size_t n = 1<<20; int bad = 0; double tc = 0, td = 0;
  unsigned char *in = malloc(n), *out = malloc(n*2+1024), *back = malloc(n);
  for(int kind=0; kind<4; kind++){
    gen(in, n, kind);
    for(int level=0; level<=9; level+= (kind==1?3:1)){
      for(int strat=0; strat<5; strat++){
        z_stream s; memset(&s,0,sizeof s);
        if(deflateInit2(&s, level, Z_DEFLATED, 15, 8, strat) != Z_OK){ printf("init fail\n"); return 1; }
        s.next_in = in; s.avail_in = n; s.next_out = out; s.avail_out = n*2+1024;
        double t0 = now(); int r = deflate(&s, Z_FINISH); tc += now()-t0;
        size_t clen = s.total_out; deflateEnd(&s);
        if(r != Z_STREAM_END){ printf("deflate failed %d\n", r); return 1; }
        uLong ad = adler32(1L, out, clen), cr = crc32(0L, out, clen);
        z_stream d; memset(&d,0,sizeof d); inflateInit(&d);
        d.next_in = out; d.avail_in = clen; d.next_out = back; d.avail_out = n;
        t0 = now(); r = inflate(&d, Z_FINISH); td += now()-t0; inflateEnd(&d);
        int ok = (r == Z_STREAM_END && d.total_out == n && memcmp(in, back, n) == 0);
        if(!ok) bad++;
        printf("kind %d level %d strat %d -> %zu adler %08lx crc %08lx %s\n", kind, level, strat, clen, ad, cr, ok ? "ok" : "ROUNDTRIP-FAIL");
      }
    }
  }
  printf("roundtrip failures: %d\n", bad);
  fprintf(stderr, "deflate %.0f ms, inflate %.0f ms\n", tc*1000, td*1000);
  return bad != 0;
}
