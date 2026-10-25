// SPDX-License-Identifier: GPL-2.0-only WITH LicenseRef-Kaiba-Runtime-Exception
//! Differential tests: C -> clang -> .ll -> Kaiba (.ll frontend, optional -O1,
//! x86_64 backend) must behave exactly like the same C compiled by gcc.
//! Skipped automatically when clang or gcc is missing.

use llcr::PassManager;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

static N: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone, Copy, PartialEq, Eq)]
enum Target { X86, A64 }

impl Target {
    fn name(self) -> &'static str { match self { Target::X86 => "x86", Target::A64 => "a64" } }
    fn cc(self) -> &'static str { match self { Target::X86 => "gcc", Target::A64 => "aarch64-linux-gnu-gcc" } }
    fn available(self) -> bool {
        have("clang") && have(self.cc()) && (self == Target::X86 || have("qemu-aarch64"))
    }
    fn clang_args(self) -> &'static [&'static str] {
        match self { Target::X86 => &[], Target::A64 => &["--target=aarch64-linux-gnu"] }
    }
    fn run(self, exe: &Path) -> std::process::Output {
        match self {
            Target::X86 => Command::new(exe).output().unwrap(),
            Target::A64 => Command::new("qemu-aarch64").args(["-L", "/usr/aarch64-linux-gnu"]).arg(exe).output().unwrap(),
        }
    }
    fn compile(self, m: &llcr::Module) -> Result<String, String> {
        match self {
            Target::X86 => kaiba_x86::compile_module(m).map_err(|e| e.to_string()),
            Target::A64 => kaiba_a64::compile_module(m).map_err(|e| e.to_string()),
        }
    }
}

fn have(tool: &str) -> bool { Command::new(tool).arg("--version").output().map(|o| o.status.success()).unwrap_or(false) }

fn scratch() -> PathBuf {
    let d = std::env::temp_dir().join(format!("kaiba-ll-{}-{}", std::process::id(), N.fetch_add(1, Ordering::SeqCst)));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn run_ok(cmd: &mut Command, what: &str) -> Vec<u8> {
    let o = cmd.output().unwrap();
    assert!(o.status.success(), "{what} failed:\n{}", String::from_utf8_lossy(&o.stderr));
    o.stdout
}

fn clang_ll(dir: &Path, c_src: &str, opt: &str, target: Target) -> String {
    let (c, ll) = (dir.join("mod.c"), dir.join("mod.ll"));
    std::fs::write(&c, c_src).unwrap();
    run_ok(
        Command::new("clang").args(target.clang_args()).args(["-S", "-emit-llvm", opt, "-fno-vectorize", "-fno-slp-vectorize", "-fno-unroll-loops", "-w", "-o"])
            .arg(&ll).arg(&c),
        "clang",
    );
    std::fs::read_to_string(ll).unwrap()
}

fn run_reference(dir: &Path, mod_c: &str, main_c: &str, target: Target) -> String {
    let (m, d, exe) = (dir.join("mod.c"), dir.join("main.c"), dir.join("ref"));
    std::fs::write(&m, mod_c).unwrap();
    std::fs::write(&d, main_c).unwrap();
    run_ok(Command::new(target.cc()).args(["-O0", "-w", "-o"]).arg(&exe).arg(&m).arg(&d).arg("-lm"), "gcc (reference)");
    let r = target.run(&exe);
    assert!(r.status.success(), "reference program failed: {:?}", r.status);
    String::from_utf8(r.stdout).unwrap()
}

fn run_kaiba(dir: &Path, ll: &str, main_c: &str, optimize: bool, tag: &str, target: Target) -> String {
    let mut m = kaiba_ll::parse_ll(ll).unwrap_or_else(|e| panic!("[{tag}] .ll parse error: {e}\n"));
    if optimize {
        let mut pm = PassManager::new();
        llcr::opt::add_o1_pipeline(&mut pm);
        pm.run(&mut m);
    }
    let asm = target.compile(&m).unwrap_or_else(|e| panic!("[{tag}] codegen error: {e}"));
    let (s, d, exe) = (dir.join("out.s"), dir.join("main.c"), dir.join("prog"));
    std::fs::write(&s, &asm).unwrap();
    std::fs::write(&d, main_c).unwrap();
    run_ok(Command::new(target.cc()).args(["-w", "-o"]).arg(&exe).arg(&d).arg(&s).arg("-lm"), &format!("[{tag}] assemble/link"));
    let r = target.run(&exe);
    assert!(r.status.success(), "[{tag}] program crashed ({:?}); asm saved at {}", r.status, {
        let keep = std::env::temp_dir().join(format!("kaiba-fail-{}.s", tag.replace([' ', '/'], "_")));
        let _ = std::fs::write(&keep, &asm);
        keep.display().to_string()
    });
    String::from_utf8(r.stdout).unwrap()
}

fn differential(name: &str, mod_c: &str, main_c: &str) {
    for target in [Target::X86, Target::A64] {
        if !target.available() { eprintln!("{} toolchain not found; skipping {name}", target.name()); continue; }
        let dir = scratch();
        let expected = run_reference(&dir, mod_c, main_c, target);
        assert!(!expected.is_empty(), "{name}: reference program printed nothing");
        for clang_opt in ["-O0", "-O1", "-O2"] {
            let ll = clang_ll(&dir, mod_c, clang_opt, target);
            for kaiba_opt in [false, true] {
                let tag = format!("{name} {} clang{clang_opt} kaiba-O{}", target.name(), kaiba_opt as u8);
                let got = run_kaiba(&dir, &ll, main_c, kaiba_opt, &tag, target);
                assert_eq!(got, expected, "[{tag}] output differs from gcc reference\n--- .ll ---\n{ll}");
            }
        }
        let _ = std::fs::remove_dir_all(dir);
    }
}

#[test]
fn arithmetic_and_loops() {
    differential("arith", r#"
int gcd(int a,int b){ while(b){int t=a%b;a=b;b=t;} return a; }
unsigned isqrt(unsigned n){ unsigned x=0; while((x+1)*(x+1)<=n) x++; return x; }
int collatz(unsigned long n){ int s=0; while(n!=1){ n=(n&1)?3*n+1:n>>1; s++; } return s; }
int popcnt(unsigned x){ int c=0; while(x){ c+=x&1; x>>=1; } return c; }
int is_prime(int n){ if(n<2) return 0; for(int i=2;i*i<=n;i++) if(n%i==0) return 0; return 1; }
long fibl(int n){ long a=0,b=1; for(int i=0;i<n;i++){ long t=a+b; a=b; b=t; } return a; }
unsigned long long fnv(const unsigned char*p,int n){ unsigned long long h=1469598103934665603ULL; for(int i=0;i<n;i++){ h^=p[i]; h*=1099511628211ULL; } return h; }
"#, r#"
#include <stdio.h>
int gcd(int,int); unsigned isqrt(unsigned); int collatz(unsigned long); int popcnt(unsigned);
int is_prime(int); long fibl(int); unsigned long long fnv(const unsigned char*,int);
int main(){
  for(int i=1;i<30;i+=4) printf("gcd %d %d\n", gcd(i*12,18), gcd(i,35));
  for(unsigned n=0;n<200;n+=37) printf("isqrt %u %u\n", n, isqrt(n));
  for(unsigned long n=1;n<40;n+=7) printf("collatz %lu %d\n", n, collatz(n));
  printf("popcnt %d %d %d\n", popcnt(0), popcnt(255), popcnt(0xdeadbeefu));
  int pc=0; for(int i=0;i<200;i++) pc+=is_prime(i); printf("primes %d\n", pc);
  printf("fib %ld %ld\n", fibl(10), fibl(80));
  unsigned char buf[32]; for(int i=0;i<32;i++) buf[i]=(unsigned char)(i*31+7);
  printf("fnv %llu\n", fnv(buf,32));
  return 0; }
"#);
}

#[test]
fn arrays_pointers_and_strings() {
    differential("arrays", r#"
void bsort(int*a,int n){ for(int i=0;i<n;i++) for(int j=0;j+1<n-i;j++) if(a[j]>a[j+1]){int t=a[j];a[j]=a[j+1];a[j+1]=t;} }
int bsearch_i(const int*a,int n,int k){ int lo=0,hi=n-1; while(lo<=hi){ int m=lo+(hi-lo)/2; if(a[m]==k) return m; if(a[m]<k) lo=m+1; else hi=m-1; } return -1; }
int my_strlen(const char*s){ int n=0; while(s[n]) n++; return n; }
int my_strcmp(const char*a,const char*b){ while(*a&&*a==*b){a++;b++;} return (unsigned char)*a-(unsigned char)*b; }
void rev(char*s){ int n=my_strlen(s); for(int i=0;i<n/2;i++){ char t=s[i]; s[i]=s[n-1-i]; s[n-1-i]=t; } }
int matsum(int k){
  int a[4][4], c[4][4];
  for(int i=0;i<4;i++) for(int j=0;j<4;j++) a[i][j]=i*4+j+k;
  for(int i=0;i<4;i++) for(int j=0;j<4;j++){ int s=0; for(int m=0;m<4;m++) s+=a[i][m]*a[m][j]; c[i][j]=s; }
  int t=0; for(int i=0;i<4;i++) for(int j=0;j<4;j++) t=t*31+c[i][j];
  return t; }
"#, r#"
#include <stdio.h>
#include <string.h>
void bsort(int*,int); int bsearch_i(const int*,int,int); int my_strlen(const char*);
int my_strcmp(const char*,const char*); void rev(char*); int matsum(int);
int main(){
  int a[12]={9,-3,7,0,12,5,-8,3,3,100,-50,1}; bsort(a,12);
  for(int i=0;i<12;i++) printf("%d ", a[i]); printf("\n");
  printf("%d %d %d\n", bsearch_i(a,12,7), bsearch_i(a,12,100), bsearch_i(a,12,4));
  printf("%d %d\n", my_strlen("hello, world"), my_strlen(""));
  int c1=my_strcmp("apple","apply"), c2=my_strcmp("b","a"), c3=my_strcmp("same","same");
  printf("%d %d %d\n", c1<0, c2>0, c3==0);
  char s[32]; strcpy(s, "kaiba compiler"); rev(s); printf("%s\n", s);
  printf("%d %d\n", matsum(0), matsum(3));
  return 0; }
"#);
}

#[test]
fn structs_pointers_switch_and_function_pointers() {
    differential("structs", r#"
struct node { int val; struct node*next; };
static struct node pool[16]; static int used;
static struct node* mk(int v){ struct node*n=&pool[used++]; n->val=v; n->next=0; return n; }
int list_sum(struct node*h){ int s=0; while(h){ s+=h->val; h=h->next; } return s; }
struct node* build(int n){ struct node*h=0; for(int i=0;i<n;i++){ struct node*x=mk(i*i); x->next=h; h=x; } return h; }
void reset(void){ used=0; }
int apply_op(int k,int a,int b){ switch(k){ case 0: return a+b; case 1: return a-b; case 2: return a*b; case 5: return a<<(b&7); default: return -1; } }
int call_through(int(*f)(int,int),int a,int b){ return f(a,b); }
int ack(int m,int n){ return m==0 ? n+1 : n==0 ? ack(m-1,1) : ack(m-1, ack(m,n-1)); }
int counter; int next_id(void){ return ++counter; }
static const int primes[8]={2,3,5,7,11,13,17,19};
int prime_at(int i){ return primes[i&7]; }
struct S { char a; short b; int c; long d; };
long s_fill(struct S*s,int x){ s->a=x; s->b=x*2; s->c=x*3; s->d=(long)x*4; return s->a+s->b+s->c+s->d; }
"#, r#"
#include <stdio.h>
struct node; struct S { char a; short b; int c; long d; };
struct node* build(int); int list_sum(struct node*); void reset(void);
int apply_op(int,int,int); int call_through(int(*)(int,int),int,int); int ack(int,int);
int next_id(void); extern int counter; int prime_at(int); long s_fill(struct S*,int);
static int sq(int a,int b){ return a*a+b; }
int main(){
  for(int n=0;n<6;n++){ reset(); printf("list %d %d\n", n, list_sum(build(n))); }
  for(int k=0;k<7;k++) printf("op %d %d\n", k, apply_op(k, 12, 3));
  printf("call %d\n", call_through(sq, 7, 3));
  printf("ack %d %d\n", ack(2,3), ack(3,3));
  next_id(); next_id(); printf("ids %d %d\n", next_id(), counter);
  for(int i=0;i<10;i+=3) printf("prime %d\n", prime_at(i));
  struct S s; printf("S %ld %d %d\n", s_fill(&s,-5), s.a, s.b);
  return 0; }
"#);
}

#[test]
fn narrow_and_wide_integers() {
    differential("ints", r#"
int narrow(int x){ signed char c=(signed char)x; short s=(short)(x*3); unsigned char u=(unsigned char)(x+200); return c + s*2 + u*3 + (c<0) + (u>100); }
unsigned wrap(unsigned a,unsigned b){ return (a-b)*(a+b) ^ (a<<5) ^ (b>>3); }
long long wide(long long a,long long b){ return (a*b) - (a>>7) + (b<<3) + a/ (b|1) + a%(b|1); }
int divs(int a,int b){ return a/b + a%b*7; }
unsigned udivs(unsigned a,unsigned b){ return a/b + a%b*7; }
int signs(int x){ return (x>>3) + (int)((unsigned)x>>3) + (x<0 ? -1 : 1); }
long ext(int x){ return (long)x + (long)(unsigned)x + (long)(short)x + (long)(unsigned char)x; }
int bits(unsigned x){ x = x - ((x>>1)&0x55555555u); x = (x&0x33333333u)+((x>>2)&0x33333333u); x=(x+(x>>4))&0x0f0f0f0fu; return (x*0x01010101u)>>24; }
"#, r#"
#include <stdio.h>
int narrow(int); unsigned wrap(unsigned,unsigned); long long wide(long long,long long);
int divs(int,int); unsigned udivs(unsigned,unsigned); int signs(int); long ext(int); int bits(unsigned);
int main(){
  int xs[]={-300,-129,-128,-1,0,1,55,127,128,255,256,70000};
  for(int i=0;i<12;i++) printf("n %d %d %ld %d %d\n", narrow(xs[i]), signs(xs[i]), ext(xs[i]), bits((unsigned)xs[i]), divs(xs[i], 7));
  for(unsigned a=1;a<400000000u;a=a*7+3) printf("w %u %u\n", wrap(a,a^12345u), udivs(a, 13));
  long long vs[]={-1000000007LL,-5,0,3,99999999999LL};
  for(int i=0;i<5;i++) for(int j=0;j<5;j++) printf("L %lld\n", wide(vs[i],vs[j]));
  return 0; }
"#);
}

#[test]
fn builtins_memcpy_tables_and_idioms() {
    differential("idioms", r#"
int clz32(unsigned x){ return x ? __builtin_clz(x) : 32; }
int ctz64(unsigned long x){ return x ? __builtin_ctzl(x) : 64; }
unsigned bs32(unsigned x){ return __builtin_bswap32(x); }
unsigned long bs64(unsigned long x){ return __builtin_bswap64(x); }
unsigned rotl(unsigned x,int k){ return (x<<(k&31)) | (x>>((32-k)&31)); }
unsigned long rotr64(unsigned long x,int k){ return (x>>(k&63)) | (x<<((64-k)&63)); }
int imin(int a,int b){ return a<b?a:b; }
unsigned umax(unsigned a,unsigned b){ return a>b?a:b; }
int iabs(int x){ return x<0?-x:x; }
int sw(int x){ switch(x){ case 0: return 10; case 1: return 20; case 2: return 35; case 3: return 7; case 4: return 99; case 5: return 1; default: return -1; } }
struct big { int a[20]; char tag; };
int copy_big(int seed){ struct big x; for(int i=0;i<20;i++) x.a[i]=seed+i; x.tag=seed; struct big y=x; y.a[3]=-1; int s=0; for(int i=0;i<20;i++) s+=y.a[i]*(i+1); return s+y.tag; }
int zinit(int k){ int buf[40]={0}; buf[k%40]=k; int s=0; for(int i=0;i<40;i++) s+=buf[i]; return s; }
int arrinit(int k){ int t[6]={5,4,3,2,1,0}; return t[k%6]*10 + t[(k+1)%6]; }
const char *dummy(void){ return "x"; }
"#, r#"
#include <stdio.h>
int clz32(unsigned); int ctz64(unsigned long); unsigned bs32(unsigned); unsigned long bs64(unsigned long);
unsigned rotl(unsigned,int); unsigned long rotr64(unsigned long,int); int imin(int,int); unsigned umax(unsigned,unsigned);
int iabs(int); int sw(int); int copy_big(int); int zinit(int); int arrinit(int);
int main(){
  unsigned us[]={0,1,2,255,65536,0x80000000u,0xdeadbeefu};
  for(int i=0;i<7;i++) printf("%d %u %u %u\n", clz32(us[i]), bs32(us[i]), rotl(us[i],5), rotl(us[i],0));
  unsigned long ls[]={0,1,256,0x8000000000000000UL,0x123456789abcdef0UL};
  for(int i=0;i<5;i++) printf("%d %lu %lu %lu\n", ctz64(ls[i]), bs64(ls[i]), rotr64(ls[i],13), rotr64(ls[i],0));
  for(int a=-3;a<=3;a+=3) for(int b=-2;b<=2;b+=2) printf("%d %u %d\n", imin(a,b), umax(a,b), iabs(a-b));
  for(int x=-1;x<8;x++) printf("sw %d\n", sw(x));
  for(int s=-2;s<4;s++) printf("big %d\n", copy_big(s*37));
  for(int k=0;k<45;k+=11) printf("z %d %d\n", zinit(k), arrinit(k));
  return 0; }
"#);
}

#[test]
fn sha256_crc32_and_quicksort() {
    differential("hashes", r#"
typedef unsigned int u32; typedef unsigned char u8; typedef unsigned long long u64;
static const u32 K[64]={
0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
#define ROR(x,n) (((x)>>(n))|((x)<<(32-(n))))
static void block(u32 h[8], const u8 *p){
  u32 w[64];
  for(int i=0;i<16;i++) w[i]=(u32)p[4*i]<<24|(u32)p[4*i+1]<<16|(u32)p[4*i+2]<<8|p[4*i+3];
  for(int i=16;i<64;i++){ u32 s0=ROR(w[i-15],7)^ROR(w[i-15],18)^(w[i-15]>>3); u32 s1=ROR(w[i-2],17)^ROR(w[i-2],19)^(w[i-2]>>10); w[i]=w[i-16]+s0+w[i-7]+s1; }
  u32 a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
  for(int i=0;i<64;i++){
    u32 S1=ROR(e,6)^ROR(e,11)^ROR(e,25), ch=(e&f)^(~e&g), t1=hh+S1+ch+K[i]+w[i];
    u32 S0=ROR(a,2)^ROR(a,13)^ROR(a,22), mj=(a&b)^(a&c)^(b&c), t2=S0+mj;
    hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2; }
  h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=hh; }
void sha256(const u8*msg,u64 len,u8 out[32]){
  u32 h[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
  u8 buf[128]; u64 i=0;
  for(; i+64<=len; i+=64) block(h,msg+i);
  u64 rem=len-i; for(u64 j=0;j<rem;j++) buf[j]=msg[i+j];
  buf[rem]=0x80; u64 total=(rem<56)?64:128;
  for(u64 j=rem+1;j<total;j++) buf[j]=0;
  u64 bits=len*8; for(int j=0;j<8;j++) buf[total-1-j]=(u8)(bits>>(8*j));
  for(u64 j=0;j<total;j+=64) block(h,buf+j);
  for(int j=0;j<8;j++){ out[4*j]=h[j]>>24; out[4*j+1]=h[j]>>16; out[4*j+2]=h[j]>>8; out[4*j+3]=h[j]; } }
u32 crc32(const u8*p,int n){ u32 table[256]; for(u32 i=0;i<256;i++){ u32 c=i; for(int k=0;k<8;k++) c=(c&1)?0xEDB88320u^(c>>1):c>>1; table[i]=c; }
  u32 c=0xFFFFFFFFu; for(int i=0;i<n;i++) c=table[(c^p[i])&0xFF]^(c>>8); return ~c; }
void qsort_i(int*a,int lo,int hi){ if(lo>=hi) return; int p=a[(lo+hi)/2],i=lo,j=hi; while(i<=j){ while(a[i]<p)i++; while(a[j]>p)j--; if(i<=j){ int t=a[i];a[i]=a[j];a[j]=t;i++;j--; } } qsort_i(a,lo,j); qsort_i(a,i,hi); }
"#, r#"
#include <stdio.h>
typedef unsigned int u32; typedef unsigned char u8; typedef unsigned long long u64;
void sha256(const u8*,u64,u8*); u32 crc32(const u8*,int); void qsort_i(int*,int,int);
int main(){
  const char* msgs[]={"","abc","The quick brown fox jumps over the lazy dog","0123456789012345678901234567890123456789012345678901234567890123456789"};
  for(int m=0;m<4;m++){ int n=0; while(msgs[m][n]) n++; u8 d[32]; sha256((const u8*)msgs[m], n, d);
    for(int i=0;i<32;i++) printf("%02x", d[i]); printf("\n"); printf("crc %08x\n", crc32((const u8*)msgs[m], n)); }
  int a[200]; unsigned s=12345; for(int i=0;i<200;i++){ s=s*1103515245u+12345u; a[i]=(int)(s>>8)%1000-500; }
  qsort_i(a,0,199); long chk=0; int ok=1; for(int i=0;i<200;i++){ chk=chk*31+a[i]; if(i&&a[i-1]>a[i]) ok=0; }
  printf("sorted %d %ld\n", ok, chk);
  return 0; }
"#);
}

#[test]
fn floating_point_programs() {
    differential("floats", r#"
#include <math.h>
#include <stdio.h>
double integrate(int n){ double h=1.0/n, s=0; for(int i=0;i<n;i++){ double x=(i+0.5)*h; s+=4.0/(1.0+x*x); } return s*h; }
double newton_sqrt(double a){ double x=a>1?a:1; for(int i=0;i<30;i++) x=0.5*(x+a/x); return x; }
double horner(const double*c,int n,double x){ double r=0; for(int i=n-1;i>=0;i--) r=r*x+c[i]; return r; }
float favg(const float*p,int n){ float s=0; for(int i=0;i<n;i++) s+=p[i]; return s/n; }
int fcmp_mix(double a,double b){ return (a<b)*1 + (a<=b)*2 + (a==b)*4 + (a!=b)*8 + (a>b)*16 + (a>=b)*32; }
long trunc_round(double x){ return (long)x + (long)(x*2.0f); }
double mixed(int i,float f,double d,long l,unsigned u){ return i*f + d/(l?l:1) - u; }
struct V { double x,y,z; };
double dot(const struct V*a,const struct V*b){ return a->x*b->x + a->y*b->y + a->z*b->z; }
double vlen(const struct V*a){ return sqrt(dot(a,a)); }
double absd(double x){ return fabs(x); }
float absf(float x){ return fabsf(x); }
double fm(double a,double b,double c){ return a*b+c; }
double negate(double x){ return -x; }
double fl(double x){ return floor(x)+ceil(x)+trunc(x); }
double powmod(double x){ return pow(x,3.0)+fmod(x,1.5); }
unsigned f2u(double x){ return (unsigned)x; }
long long f2ll(float x){ return (long long)x; }
double u2d(unsigned long x){ return (double)x; }
float d2f(double x){ return (float)x; }
static const double tab[4]={0.5,1.25,-3.0,100.0};
double tabsum(int k){ double s=0; for(int i=0;i<4;i++) s+=tab[(i+k)&3]*(i+1); return s; }
void show(double d,int i,float f){ printf("show %.3f %d %.2f\n", d, i, f); }
"#, r#"
#include <stdio.h>
#include <math.h>
struct V { double x,y,z; };
double integrate(int); double newton_sqrt(double); double horner(const double*,int,double); float favg(const float*,int);
int fcmp_mix(double,double); long trunc_round(double); double mixed(int,float,double,long,unsigned);
double dot(const struct V*,const struct V*); double vlen(const struct V*); double absd(double); float absf(float);
double fm(double,double,double); double negate(double); double fl(double); double powmod(double);
unsigned f2u(double); long long f2ll(float); double u2d(unsigned long); float d2f(double); double tabsum(int);
void show(double,int,float);
int main(){
  printf("%.12f %.12f\n", integrate(10), integrate(1000));
  printf("%.12f %.12f\n", newton_sqrt(2.0), newton_sqrt(1e6));
  double c[]={1.5,-2.0,0.25,3.0}; printf("%.9g %.9g\n", horner(c,4,2.0), horner(c,4,-1.5));
  float fs[]={1.5f,2.25f,-0.75f,10.0f}; printf("%.6f\n", favg(fs,4));
  double vs[]={-1.0,0.0,2.5,2.5}; for(int i=0;i<3;i++) printf("cmp %d\n", fcmp_mix(vs[i],vs[i+1]));
  printf("%ld %ld\n", trunc_round(7.9), trunc_round(-3.3));
  printf("%.9g\n", mixed(3, 1.5f, 10.0, 4, 2));
  struct V a={1,2,3}, b={-4,5.5,0.25}; printf("%.9g %.9g\n", dot(&a,&b), vlen(&a));
  printf("%g %g %g %g %g\n", absd(-2.5), absd(3.0), absf(-0.5f), fm(2,3,4), negate(1.25));
  printf("%g %g\n", fl(2.5), fl(-2.5));
  printf("%.9g %.9g\n", powmod(2.0), powmod(-2.75));
  printf("%u %lld %.1f %.1f %.3f\n", f2u(3000000000.0), f2ll(-123456.75f), u2d(18446744073709551615UL), u2d(5), d2f(0.1));
  for(int k=0;k<4;k++) printf("%.6g\n", tabsum(k));
  show(3.14159, 7, 2.5f);
  return 0; }
"#);
}

#[test]
fn pointer_tables_and_relocations() {
    differential("relocs", r#"
static const char *names[] = {"alpha","beta","gamma","delta"};
const char *name_of(int i){ return names[i & 3]; }
static int add(int a,int b){ return a+b; }
static int sub(int a,int b){ return a-b; }
static int mul(int a,int b){ return a*b; }
typedef int (*op_t)(int,int);
static const op_t ops[3] = {add, sub, mul};
int run_op(int k,int a,int b){ return ops[k%3](a,b); }
struct Entry { const char *key; int val; };
static const struct Entry table[] = {{"one",1},{"two",2},{"three",3}};
int lookup(const char *k){
  for(int i=0;i<3;i++){ const char *a=table[i].key,*b=k; while(*a && *a==*b){a++;b++;} if(*a==*b) return table[i].val; }
  return -1; }
static int arr[8] = {1,2,3,4,5,6,7,8};
int *ptr_into = &arr[3];
int *end_ptr = arr + 8;
int sum_range(void){ int s=0; for(int *p=ptr_into;p<end_ptr;p++) s+=*p; return s; }
int bump_arr(int i){ arr[i&7]++; return arr[i&7]; }
const char *const *names_ptr = names;
"#, r#"
#include <stdio.h>
const char *name_of(int); int run_op(int,int,int); int lookup(const char*); int sum_range(void); int bump_arr(int);
extern const char *const *names_ptr;
int main(){
  for(int i=0;i<6;i++) printf("%s ", name_of(i)); printf("\n");
  for(int k=0;k<6;k++) printf("%d ", run_op(k, 17, 5)); printf("\n");
  printf("%d %d %d %d\n", lookup("one"), lookup("three"), lookup("two"), lookup("nope"));
  printf("%d\n", sum_range());
  bump_arr(4); bump_arr(4); printf("%d %d\n", bump_arr(4), sum_range());
  printf("%s %s\n", names_ptr[1], names_ptr[3]);
  return 0; }
"#);
}

#[test]
fn overflow_builtins() {
    differential("overflow", r#"
#include <limits.h>
int  sadd(int a,int b,int*r){ return __builtin_add_overflow(a,b,r); }
int  ssub(int a,int b,int*r){ return __builtin_sub_overflow(a,b,r); }
int  smul(int a,int b,int*r){ return __builtin_mul_overflow(a,b,r); }
int  uadd(unsigned a,unsigned b,unsigned*r){ return __builtin_add_overflow(a,b,r); }
int  usub(unsigned a,unsigned b,unsigned*r){ return __builtin_sub_overflow(a,b,r); }
int  umul(unsigned a,unsigned b,unsigned*r){ return __builtin_mul_overflow(a,b,r); }
int  smul64(long a,long b,long*r){ return __builtin_mul_overflow(a,b,r); }
int  umul64(unsigned long a,unsigned long b,unsigned long*r){ return __builtin_mul_overflow(a,b,r); }
int  sadd64(long a,long b,long*r){ return __builtin_add_overflow(a,b,r); }
int  smul16(short a,short b,short*r){ return __builtin_mul_overflow(a,b,r); }
int  uadd8(unsigned char a,unsigned char b,unsigned char*r){ return __builtin_add_overflow(a,b,r); }
int checked_sum(const int*p,int n,int*out){ int s=0; for(int i=0;i<n;i++) if(__builtin_add_overflow(s,p[i],&s)) return 1; *out=s; return 0; }
"#, r#"
#include <stdio.h>
#include <limits.h>
int sadd(int,int,int*); int ssub(int,int,int*); int smul(int,int,int*); int uadd(unsigned,unsigned,unsigned*);
int usub(unsigned,unsigned,unsigned*); int umul(unsigned,unsigned,unsigned*); int smul64(long,long,long*);
int umul64(unsigned long,unsigned long,unsigned long*); int sadd64(long,long,long*); int smul16(short,short,short*);
int uadd8(unsigned char,unsigned char,unsigned char*); int checked_sum(const int*,int,int*);
int main(){
  int is[]={0,1,-1,2,-2,46341,-46341,65536,INT_MAX,INT_MIN,INT_MAX-1,INT_MIN+1,100000};
  int n=sizeof is/sizeof is[0];
  for(int i=0;i<n;i++) for(int j=0;j<n;j++){ int a=is[i],b=is[j],r;
    int f1=sadd(a,b,&r); int r1=r; int f2=ssub(a,b,&r); int r2=r; int f3=smul(a,b,&r);
    printf("%d %d %d %d %d %d\n", f1,r1,f2,r2,f3,r); }
  unsigned us[]={0,1,2,65535,65536,4294967295u,2147483648u,3000000000u};
  for(int i=0;i<8;i++) for(int j=0;j<8;j++){ unsigned r;
    int f1=uadd(us[i],us[j],&r); unsigned r1=r; int f2=usub(us[i],us[j],&r); unsigned r2=r; int f3=umul(us[i],us[j],&r);
    printf("%d %u %d %u %d %u\n", f1,r1,f2,r2,f3,r); }
  long ls[]={0,1,-1,3037000499L,-3037000499L,3037000500L,LONG_MAX,LONG_MIN,4294967296L,-4294967296L};
  for(int i=0;i<10;i++) for(int j=0;j<10;j++){ long r; int f1=smul64(ls[i],ls[j],&r); long r1=r; int f2=sadd64(ls[i],ls[j],&r);
    printf("%d %ld %d %ld\n", f1,r1,f2,r); }
  unsigned long uls[]={0,1,4294967295UL,4294967296UL,18446744073709551615UL,9223372036854775808UL};
  for(int i=0;i<6;i++) for(int j=0;j<6;j++){ unsigned long r; int f=umul64(uls[i],uls[j],&r); printf("%d %lu\n", f, r); }
  short ss[]={0,1,-1,181,182,-182,32767,-32768}; for(int i=0;i<8;i++) for(int j=0;j<8;j++){ short r; int f=smul16(ss[i],ss[j],&r); printf("%d %d\n", f, r); }
  unsigned char bs[]={0,1,127,128,200,255}; for(int i=0;i<6;i++) for(int j=0;j<6;j++){ unsigned char r; int f=uadd8(bs[i],bs[j],&r); printf("%d %d\n", f, r); }
  int arr1[]={1,2,3,4}, arr2[]={INT_MAX,1}, out=0; printf("%d %d\n", checked_sum(arr1,4,&out), out); printf("%d\n", checked_sum(arr2,2,&out));
  return 0; }
"#);
}

#[test]
fn large_frames_many_args_wide_immediates_and_char_signedness() {
    differential("abi-edges", r#"
long bigframe(int k){ char buf[20000]; for(int i=0;i<20000;i++) buf[i]=(char)(i*7+k); long s=0; for(int i=0;i<20000;i+=97) s+=buf[i]; return s; }
long manyargs(long a,long b,long c,long d,long e,long f,long g,long h,long i,long j,long k,long l){
  return a+2*b+3*c+4*d+5*e+6*f+7*g+8*h+9*i+10*j+11*k+12*l; }
long callmany(long x){ return manyargs(x,x+1,x+2,x+3,x+4,x+5,x+6,x+7,x+8,x+9,x+10,x+11); }
double mixmany(int a,double b,int c,double d,int e,double f,int g,double h,int i,double j,int k,double l,int m,double n,int o,double p,int q,double r,int s){
  return a+b+c+d+e+f+g+h+i+j+k+l+m+n+o+p+q+r+s; }
double callmix(double x){ return mixmany(1,x,3,x*2,5,x*3,7,x*4,9,x*5,11,x*6,13,x*7,15,x*8,17,x*9,19); }
unsigned long bigconst(unsigned long x){ return x ^ 0x123456789abcdef0UL ^ 0xffff0000ffff0000UL ^ 0xfffffffffffffffeUL ^ 0x0000ffff00000000UL ^ 0x00000000ffff0000UL; }
long negconst(long x){ return x + (-1) + (-65536) + (-4294967296L) + 0xffffffffL; }
int signed_char_sum(const char*s,int n){ int t=0; for(int i=0;i<n;i++) t+=s[i]; return t; }
int uchar_sum(const unsigned char*s,int n){ int t=0; for(int i=0;i<n;i++) t+=s[i]; return t; }
int spills(int a,int b){ int v0=a+1,v1=b+2,v2=a*3,v3=b*5,v4=a^b,v5=a-b,v6=a|b,v7=a&b,v8=a<<2,v9=b>>1,v10=a+b,v11=a*b,v12=v0+v1,v13=v2^v3,v14=v4+v5,v15=v6-v7,v16=v8*v9,v17=v10+v11;
  return v0+v1*2+v2*3+v3*4+v4*5+v5*6+v6*7+v7*8+v8*9+v9*10+v10*11+v11*12+v12*13+v13*14+v14*15+v15*16+v16*17+v17*18; }
"#, r#"
#include <stdio.h>
long bigframe(int); long callmany(long); double callmix(double); unsigned long bigconst(unsigned long); long negconst(long);
int signed_char_sum(const char*,int); int uchar_sum(const unsigned char*,int); int spills(int,int);
int main(){
  for(int k=0;k<4;k++) printf("big %ld\n", bigframe(k*13));
  printf("many %ld %ld\n", callmany(0), callmany(100));
  printf("mix %.3f\n", callmix(1.5));
  printf("const %lu %lu\n", bigconst(0), bigconst(0xdeadbeefcafef00dUL));
  printf("neg %ld %ld\n", negconst(0), negconst(123456789012L));
  unsigned char bytes[8]={1,127,128,200,255,0,90,180};
  printf("chars %d %d\n", signed_char_sum((const char*)bytes,8), uchar_sum(bytes,8));
  for(int i=-3;i<4;i++) printf("spill %d\n", spills(i*17, i*i+3));
  return 0; }
"#);
}

#[test]
fn struct_arguments_and_returns_in_both_directions() {
    differential("struct-abi", r#"
typedef struct { long q, r; } divres;
typedef struct { double x, y; } vec2;
typedef struct { long a, b, c, d; } big;
typedef struct { long l; double d; } mixed;
typedef struct { int a, b; } pair;
typedef struct { char c; short s; int i; long l; } packed_mix;
divres divmod(long a, long b){ divres d = { a / b, a % b }; return d; }
vec2 vadd(vec2 a, vec2 b){ vec2 r = { a.x + b.x, a.y + b.y }; return r; }
vec2 vscale(vec2 a, double k){ vec2 r = { a.x * k, a.y * k }; return r; }
big mkbig(long x){ big b = { x, x + 1, x + 2, x + 3 }; return b; }
long sumbig(big b){ return b.a + b.b * 2 + b.c * 3 + b.d * 4; }
big bigadd(big a, big b){ big r = { a.a + b.a, a.b + b.b, a.c + b.c, a.d + b.d }; return r; }
mixed mk(long l, double d){ mixed m = { l, d }; return m; }
double mixsum(mixed m){ return m.l * 10 + m.d; }
pair swap_pair(pair p){ pair q = { p.b, p.a }; return q; }
packed_mix pm_make(int i){ packed_mix p = { (char)i, (short)(i * 3), i * 5, (long)i * 7 }; return p; }
long pm_sum(packed_mix p){ return p.c + p.s + p.i + p.l; }
extern divres ext_divmod(long, long);
extern big ext_big(long);
extern long ext_sumbig(big);
extern mixed ext_mixed(long, double);
long roundtrip(long x){ divres d = ext_divmod(x, 7); big b = ext_big(d.q); mixed m = ext_mixed(d.r, 0.5); return d.r + ext_sumbig(b) + sumbig(b) + (long)mixsum(m); }
big pass_through(big b){ return bigadd(b, mkbig(1)); }
vec2 vloop(int n){ vec2 v = { 0, 0 }; for (int i = 0; i < n; i++) { vec2 d = { i, i * 0.5 }; v = vadd(v, d); } return v; }
long chain(long x){ divres d = divmod(x, 5); pair p = { (int)d.q, (int)d.r }; pair s = swap_pair(p); packed_mix pm = pm_make(s.a); return pm_sum(pm) + s.b + sumbig(mkbig(s.a)); }
"#, r#"
#include <stdio.h>
typedef struct { long q, r; } divres;
typedef struct { double x, y; } vec2;
typedef struct { long a, b, c, d; } big;
typedef struct { long l; double d; } mixed;
typedef struct { int a, b; } pair;
typedef struct { char c; short s; int i; long l; } packed_mix;
divres divmod(long,long); vec2 vadd(vec2,vec2); vec2 vscale(vec2,double); big mkbig(long); long sumbig(big);
big bigadd(big,big); mixed mk(long,double); double mixsum(mixed); pair swap_pair(pair);
packed_mix pm_make(int); long pm_sum(packed_mix); long roundtrip(long); big pass_through(big);
vec2 vloop(int); long chain(long);
divres ext_divmod(long a, long b){ divres d = { a / b, a % b }; return d; }
big ext_big(long x){ big b = { x * 2, x * 3, x * 4, x * 5 }; return b; }
long ext_sumbig(big b){ return b.a - b.b + b.c - b.d; }
mixed ext_mixed(long l, double d){ mixed m = { l + 1, d * 2 }; return m; }
int main(){
  divres d = divmod(100, 7); printf("divmod %ld %ld\n", d.q, d.r);
  vec2 a = {1.5, 2.5}, b = {10, 20}; vec2 s = vadd(a, b); vec2 k = vscale(s, 2.0); printf("vec %.3f %.3f %.3f %.3f\n", s.x, s.y, k.x, k.y);
  big g = mkbig(5); printf("big %ld %ld %ld %ld sum %ld\n", g.a, g.b, g.c, g.d, sumbig(g));
  big h = bigadd(g, mkbig(100)); printf("bigadd %ld %ld %ld %ld\n", h.a, h.b, h.c, h.d);
  mixed m = mk(7, 2.5); printf("mixed %ld %.2f %.2f\n", m.l, m.d, mixsum(m));
  pair p = {3, 9}; pair q = swap_pair(p); printf("pair %d %d\n", q.a, q.b);
  packed_mix pm = pm_make(11); printf("pm %d %d %d %ld sum %ld\n", pm.c, pm.s, pm.i, pm.l, pm_sum(pm));
  for (int i = 0; i < 3; i++) printf("roundtrip %ld\n", roundtrip(50 + i * 13));
  big t = pass_through(g); printf("pass %ld %ld %ld %ld\n", t.a, t.b, t.c, t.d);
  vec2 v = vloop(10); printf("vloop %.3f %.3f\n", v.x, v.y);
  for (int i = 0; i < 4; i++) printf("chain %ld\n", chain(17 + i * 9));
  return 0; }
"#);
}

#[test]
fn variadic_function_definitions() {
    differential("varargs", r#"
#include <stdarg.h>
#include <stdio.h>
int sum_ints(int n, ...){ va_list ap; va_start(ap, n); int s = 0; for (int i = 0; i < n; i++) s += va_arg(ap, int) * (i + 1); va_end(ap); return s; }
double sum_doubles(int n, ...){ va_list ap; va_start(ap, n); double s = 0; for (int i = 0; i < n; i++) s += va_arg(ap, double); va_end(ap); return s; }
long mixed(int n, ...){
  va_list ap; va_start(ap, n); long acc = 0;
  for (int i = 0; i < n; i++) {
    switch (i % 3) { case 0: acc += va_arg(ap, long); break; case 1: acc += (long)(va_arg(ap, double) * 10); break; default: acc += va_arg(ap, int); }
  }
  va_end(ap); return acc; }
int vformat(char *buf, int size, const char *fmt, ...){ va_list ap; va_start(ap, fmt); int r = vsnprintf(buf, size, fmt, ap); va_end(ap); return r; }
int copy_test(int n, ...){ va_list a, b; va_start(a, n); va_copy(b, a); int s1 = 0, s2 = 0;
  for (int i = 0; i < n; i++) s1 += va_arg(a, int);
  for (int i = 0; i < n; i++) s2 += va_arg(b, int) * 2;
  va_end(a); va_end(b); return s1 * 1000 + s2; }
long named_then_va(long a, double b, int c, ...){ va_list ap; va_start(ap, c); long s = a + (long)b + c; for (int i = 0; i < 3; i++) s += va_arg(ap, long) * (i + 2); va_end(ap); return s; }
int callvar(void){ return sum_ints(8, 1, 2, 3, 4, 5, 6, 7, 8) + (int)sum_doubles(10, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0) + (int)mixed(7, 1L, 2.5, 3, 4L, 5.5, 6, 7L); }
"#, r#"
#include <stdio.h>
int sum_ints(int, ...); double sum_doubles(int, ...); long mixed(int, ...); int vformat(char*, int, const char*, ...);
int copy_test(int, ...); long named_then_va(long, double, int, ...); int callvar(void);
int main(){
  printf("%d %d %d\n", sum_ints(0), sum_ints(1, 5), sum_ints(3, 10, 20, 30));
  printf("%d\n", sum_ints(12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12));
  printf("%.2f %.2f\n", sum_doubles(3, 1.5, 2.5, 3.5), sum_doubles(12, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0));
  printf("%ld %ld\n", mixed(3, 100L, 1.5, 7), mixed(9, 1L, 1.0, 2, 3L, 2.0, 4, 5L, 3.0, 6));
  char buf[128]; int n = vformat(buf, sizeof buf, "%d-%s-%.2f-%c-%ld", 42, "str", 3.14159, 'x', 123456789012L); printf("%d [%s]\n", n, buf);
  printf("%d %d\n", copy_test(3, 1, 2, 3), copy_test(5, 5, 4, 3, 2, 1));
  printf("%ld %ld\n", named_then_va(1, 2.9, 3, 10L, 20L, 30L), named_then_va(-5, 100.5, 7, 1L, 1L, 1L));
  printf("%d\n", callvar());
  return 0; }
"#);
}

#[test]
fn addressing_mode_edge_cases() {
    differential("addressing", r#"
#include <stdlib.h>
#include <string.h>
struct S12 { int a, b, c; };
struct Mixed { char c; long l; short s; double d; int i[5]; };
long field_far(char *p){ return *(long*)(p + 100000) + *(int*)(p - 77) + *(short*)(p + 4097) + *(char*)(p + 255) + *(char*)(p - 256) + *(long*)(p + 32760) + *(long*)(p + 32768); }
long never_called_huge(char *p){ return *(char*)(p + 4294967296L) + *(int*)(p - 4294967296L); }
long idx_scales(int *a, short *s, char *c, long *l, double *d, int i){ return a[i] + s[i] + c[i] + l[i] + (long)d[i]; }
int idx12(struct S12 *p, int i){ return p[i].c * 100 + p[i].a * 10 + p[i+1].b; }
int neg_idx(int *a, long i){ return a[-i] + a[i - 3] + a[-i - 1]; }
long mixed_fields(struct Mixed *m, int k){ return m->c + m->l + m->s + (long)m->d + m->i[k] + m[1].i[4] + (m + 1)->l; }
int two_d(int (*g)[7], int r, int c){ return g[r][c] + g[r + 1][c - 1] + g[0][6]; }
unsigned char byte_scan(const unsigned char *p, int n){ unsigned char x = 0; for (int i = 0; i < n; i++) x ^= p[i] + p[n - 1 - i]; return x; }
long store_forms(long *l, int *w, short *h, char *b, int i){ l[i] = 11; w[i + 1] = 22; h[i - 1] = 33; b[i * 3] = 44; l[-1 + i] = 55; return l[i] + w[i + 1] + h[i - 1] + b[i * 3] + l[i - 1]; }
long global_table_idx(int i);
static const int tbl[16] = {3,1,4,1,5,9,2,6,5,3,5,8,9,7,9,3};
long global_table_idx(int i){ return tbl[i & 15] * 100 + tbl[(i + 5) & 15]; }
"#, r#"
#include <stdio.h>
#include <string.h>
struct S12 { int a, b, c; };
struct Mixed { char c; long l; short s; double d; int i[5]; };
long field_far(char*); long idx_scales(int*,short*,char*,long*,double*,int); int idx12(struct S12*,int); int neg_idx(int*,long);
long mixed_fields(struct Mixed*,int); int two_d(int(*)[7],int,int); unsigned char byte_scan(const unsigned char*,int);
long store_forms(long*,int*,short*,char*,int); long global_table_idx(int);
int main(){
  static char big[300000]; for(int i=0;i<300000;i++) big[i]=(char)(i*131+7);
  printf("far %ld\n", field_far(big + 120000));
  int a[16]; short s[16]; char c[16]; long l[16]; double d[16];
  for(int i=0;i<16;i++){ a[i]=i*3-7; s[i]=i*5-9; c[i]=(char)(i*11-3); l[i]=i*7-20; d[i]=i*1.5+0.25; }
  for(int i=0;i<16;i+=3) printf("scales %ld\n", idx_scales(a,s,c,l,d,i));
  struct S12 arr[6]; for(int i=0;i<6;i++){ arr[i].a=i; arr[i].b=i*2; arr[i].c=i*3; }
  for(int i=0;i<5;i++) printf("s12 %d\n", idx12(arr,i));
  int g[16]; for(int i=0;i<16;i++) g[i]=i*i-5; printf("neg %d %d\n", neg_idx(g+8,2), neg_idx(g+10,4));
  struct Mixed m[2]; memset(m,0,sizeof m); for(int j=0;j<2;j++){ m[j].c=j+3; m[j].l=j*1000+5; m[j].s=j*7+1; m[j].d=j+2.5; for(int k=0;k<5;k++) m[j].i[k]=j*10+k; }
  for(int k=0;k<5;k++) printf("mixed %ld\n", mixed_fields(m,k));
  int grid[4][7]; for(int r=0;r<4;r++) for(int q=0;q<7;q++) grid[r][q]=r*10+q; printf("2d %d %d\n", two_d(grid,0,3), two_d(grid,2,6));
  unsigned char bytes[33]; for(int i=0;i<33;i++) bytes[i]=(unsigned char)(i*37+1); printf("scan %u %u\n", byte_scan(bytes,33), byte_scan(bytes,10));
  long sl[12]; int sw[12]; short sh[12]; char sb[40]; memset(sl,0,sizeof sl); memset(sw,0,sizeof sw); memset(sh,0,sizeof sh); memset(sb,0,sizeof sb);
  for(int i=1;i<10;i+=4) printf("store %ld\n", store_forms(sl+1,sw,sh+1,sb,i));
  for(int i=0;i<20;i+=7) printf("table %ld\n", global_table_idx(i));
  return 0; }
"#);
}
