// Exact conditional central mod8 + full second-quotient zero-shift cross product.
#include <array>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <map>
#include <unordered_set>
#include <bit>
#include <cassert>
using namespace std;
using W=uint64_t;using I=uint32_t;using U=uint16_t;
constexpr int N=4096,F=N-1,Q=23;constexpr I QM=(1u<<23)-1;
constexpr W MASK=0x777777777777ULL;
array<I,N> expand{};array<int,N> mass{},sq{};vector<array<U,4>> supports;
I rot(I x,int t){return ((x<<t)|(x>>(Q-t)))&QM;}
void init(){for(int x=0;x<N;x++){for(int j=0;j<12;j++)if(x>>j&1){expand[x]|=1u<<j;if(j)expand[x]|=1u<<(Q-j);}mass[x]=popcount(expand[x]);for(int j=0;j<12;j++)if(x>>j&1){int k=2*j%23;sq[x]|=1<<min(k,23-k);}}
 vector<U> mul(N);for(int s=0;s<N;s++){array<int,12>col{};for(int j=0;j<12;j++)col[j]=(j?rot(expand[s],j)^rot(expand[s],23-j):expand[s])&F;for(int w=1;w<N;w++)mul[w]=mul[w&(w-1)]^col[countr_zero(unsigned(w))];assert(mul[s]==sq[s]);for(int w=0;w<N;w++){int t=1^s^w^sq[s]^sq[w]^mul[w];array<U,4> a{U(t),U(t^s),U(t^w),U(t^s^w^1)};if((mass[a[0]]+mass[a[1]]+mass[a[2]]+mass[a[3]])%8==7)supports.push_back(a);}}
 assert(supports.size()==8392704);}
struct Key{W u,v;int d;bool operator==(const Key&)const=default;bool operator<(const Key&o)const{return u!=o.u?u<o.u:v!=o.v?v<o.v:d<o.d;}};
Key add(Key a,Key b){return {(a.u+b.u)&MASK,(a.v+b.v)&MASK,a.d+b.d};}
Key sub(Key a,Key b){return {(a.u+(MASK^b.u)+0x111111111111ULL)&MASK,(a.v+(MASK^b.v)+0x111111111111ULL)&MASK,a.d-b.d};}
struct Hash{size_t operator()(Key a)const{return a.u*0x9e3779b97f4a7c15ULL+rotl(a.v,23)+uint64_t(a.d)*0xbf58476d1ce4e5b9ULL;}};
W sig(int p,int n){W key=0;I a=expand[p],b=expand[n];for(int t=0;t<12;t++){int r=popcount(a&rot(a,t))+popcount(b&rot(b,t))-popcount(a&rot(b,t))-popcount(b&rot(a,t));key|=W((r%8+8)%8)<<(4*t);}return key;}
array<int,8> val{};vector<array<int,8>> hist;
struct Row{array<int8_t,12> a;U plus;};
void gen(int j,int p,int n,array<int,8>&h,Row x,vector<Row>&out){if(j==12){for(int z:h)assert(!z);out.push_back(x);return;}int r=(p>>j&1)?1:(n>>j&1)?2:0;for(int k=0;k<8;k++)if(h[k]&&(val[k]%3+3)%3==r){--h[k];x.a[j]=val[k];if((val[k]%8+8)%8==5)x.plus|=1<<j;else x.plus&=~(1<<j);gen(j+1,p,n,h,x,out);++h[k];}}
vector<Row> rows(int h,int s,int p,int n){assert(!(p&n));assert((s%3+3)%3==((p&1)?1:(n&1)?2:0));Row x{};x.a[0]=s;x.plus=((s%8+8)%8==5);auto a=hist[h];vector<Row> out;gen(1,p,n,a,x,out);assert(!out.empty());return out;}
using Bank=array<vector<Key>,N>;
Bank makebank(int hx,int hy,int sx,int sy,W c,W &raw){auto a=rows(hx,sx,c&F,(c>>12)&F),b=rows(hy,sy,(c>>24)&F,(c>>36)&F);Bank out;raw+=W(a.size())*b.size();for(auto &x:a)for(auto &y:b){int p=x.plus,q=y.plus;int v=p^q,up=p&q,un=F^(p|q),vp=p&~q,vn=q&~p;int d=x.a[0]*y.a[0];for(int j=1;j<12;j++)d+=2*x.a[j]*y.a[j];out[v].push_back({sig(up,un),sig(vp,vn),d});}for(auto &b:out){sort(b.begin(),b.end());b.erase(unique(b.begin(),b.end()),b.end());}return out;}
struct Job{int mult;array<int,8> h,s;};
int main(int argc,char**argv){assert(argc==5);init();ifstream h(argv[1]),in(argv[2]),wit(argv[3]);W cap=stoull(argv[4]);int nh;h>>nh;for(int &x:val)h>>x;hist.resize(nh);for(auto&a:hist)for(int&x:a)h>>x;map<int,Job> jobs;int id;Job j;while(in>>id>>j.mult){for(int&x:j.h)in>>x;for(int&x:j.s)in>>x;jobs[id]=j;}Key target{0,0,-26};for(int t=0;t<12;t++){target.u|=W((t?-26:573)&7)<<(4*t);target.v|=W((t?0:599)&7)<<(4*t);}array<W,4> codes;while(wit>>id){for(W &x:codes)wit>>x;j=jobs.at(id);W raw=0,tuples=0,work=0,keys=0,plans=0;bool hit=false,capped=false;array<Bank,4>b;for(int f=0;f<4;f++){b[f]=makebank(j.h[2*f],j.h[2*f+1],j.s[2*f],j.s[2*f+1],codes[f],raw);for(auto &v:b[f])keys+=v.size();}
 for(auto p:supports){array<const vector<Key>*,4>a;bool ok=true;for(int i=0;i<4;i++){a[i]=&b[i][p[i]];if(a[i]->empty()){ok=false;break;}}if(!ok)continue;++tuples;W best=~0ULL;array<int,4> o{};for(int i=1;i<4;i++){int k=i==1?2:1,l=6-i-k;W x=W(a[0]->size())*a[i]->size(),y=W(a[k]->size())*a[l]->size();if(x+y<best){best=x+y;o=x<=y?array<int,4>{0,i,k,l}:array<int,4>{k,l,0,i};}}plans+=best;if(hit)continue;if(work+best>cap){capped=true;continue;}unordered_set<Key,Hash> tab;for(auto x:*a[o[0]])for(auto y:*a[o[1]]){tab.insert(add(x,y));++work;}for(auto x:*a[o[2]]){for(auto y:*a[o[3]]){++work;if(tab.contains(sub(target,add(x,y)))){hit=true;break;}}if(hit)break;}}
 cout<<id<<','<<j.mult<<','<<(hit?"POS":capped?"CAP":"NEG")<<','<<raw<<','<<keys<<','<<tuples<<','<<plans<<','<<work<<endl;}
 return 0;}
