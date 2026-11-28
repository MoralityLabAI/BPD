// Exact central-column modulo-12 row signature join on Z23.
#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <map>
#include <unordered_map>
#include <vector>
using namespace std;
using u64=uint64_t;using u32=uint32_t;
constexpr int Q=23, K=12; constexpr u32 FULL=(1u<<Q)-1;
constexpr u64 rep(unsigned v){u64 z=0;for(int i=0;i<K;i++)z|=u64(v)<<(5*i);return z;}
constexpr u64 HI=rep(16),FOUR=rep(4),TWELVE=rep(12);
u64 reduce(u64 t){return t-(((t+FOUR)&HI)>>4)*12;}
u64 add(u64 a,u64 b){return reduce(a+b);}
u64 sub(u64 a,u64 b){return reduce(a+TWELVE-b);}
struct Key{u64 u,v; bool operator==(const Key&)const=default;};
struct Hash{size_t operator()(Key k)const {u64 x=k.u^(rotl(k.v,23));x^=x>>30;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27;x*=0x94d049bb133111ebULL;return x^(x>>31);}};
Key add(Key a,Key b){return {add(a.u,b.u),add(a.v,b.v)};}
Key sub(Key a,Key b){return {sub(a.u,b.u),sub(a.v,b.v)};}
u32 rot(u32 a,int t){return ((a<<t)|(a>>(Q-t)))&FULL;}
u64 sig(u32 p,u32 n){u64 s=0;for(int k=0,t=0;k<K;k++,t=k==1?1:2*t%Q) {int r=popcount(p&rot(p,t))+popcount(n&rot(n,t))-popcount(p&rot(n,t))-popcount(n&rot(p,t));s|=u64((r%12+12)%12)<<(5*k);}return s;}
using Row=array<u32,4>;
Key sig(Row r){return {sig(r[0],r[1]),sig(r[2],r[3])};}
using Bank=vector<pair<Key,Row>>;
void gen(int j,int u,int v,Row r, Bank& b,u64& count){
 if(j==K){if(u||v)return; ++count;b.emplace_back(sig(r),r);return;}
 int w=j?2:1;int rem=2*(K-j-1);u32 mask=(1u<<j)|(j?1u<<(Q-j):0);
 for(int d=0;d<4;d++){
  int a=u-(d==0?w:d==1?-w:0),c=v-(d==2?w:d==3?-w:0);
  if(abs(a)+abs(c)>rem || ((rem-abs(a)-abs(c))%4 && j>0) || (a%2!=0)||(c%2!=0))continue;
  // For remaining pairs, half-sums have parity matching remaining pair count.
  if((rem-abs(a)-abs(c))%4!=0)continue;
  Row z=r;z[d]|=mask;gen(j+1,a,c,z,b,count);
 }
}
Bank bank(int u,int v,u64& count){Bank b;count=0;gen(0,u,v,{},b,count);sort(b.begin(),b.end(),[](auto a,auto z){return a.first.u<z.first.u||(a.first.u==z.first.u&&a.first.v<z.first.v);});b.erase(unique(b.begin(),b.end(),[](auto a,auto z){return a.first==z.first;}),b.end());return b;}
constexpr u64 NONZERO=(1ULL<<55)-1;
u64 rotate_lanes(u64 a){u64 b=a>>5;return (a&31)|(((b>>5)|((b&31)<<50))<<5);}
Key rotate_key(Key a){return {rotate_lanes(a.u),rotate_lanes(a.v)};}
bool lesskey(Key a,Key b){return a.u<b.u||(a.u==b.u&&a.v<b.v);}
pair<Key,int> canon(Key a){Key best=a;int r=0;for(int k=1;k<11;k++){a=rotate_key(a);if(lesskey(a,best)){best=a;r=k;}}return {best,r};}
Row rotate_row(Row r,int k){ // f'(i)=f(2^k i), so PAF'(t)=PAF(2^k t)
 int p=1;for(int i=0;i<k;i++)p=p*2%Q;
 Row z{};for(int a=0;a<4;a++)for(int i=0;i<Q;i++)if((r[a]>>(p*i%Q))&1)z[a]|=1u<<i;
 return z;
}
using Profile=array<pair<int,int>,4>;
struct Result{int status;u64 work0,work1; array<u64,4> counts,sizes;array<Row,4> rows;};
Result solve(Profile p,Key target,u64 cap=12000000){
 Result z{};array<Bank,4> b,c;
 for(int i=0;i<4;i++){
  b[i]=bank(p[i].first,p[i].second,z.counts[i]);z.sizes[i]=b[i].size();if(b[i].empty()){z.status=0;return z;}
  for(auto a:b[i])if(a.first==canon(a.first).first)c[i].push_back(a);
 }
 assert(rotate_key(target)==target);
 array<int,4> best{0,1,2,3};u64 cost=~0ULL;
 for(int j=1;j<4;j++){
  array<int,4>a{0,j,0,0};int k=2;for(int i=1;i<4;i++)if(i!=j)a[k++]=i;
  for(int t=0;t<2;t++)if(c[a[2*t]].size()*b[a[2*t+1]].size()>c[a[2*t+1]].size()*b[a[2*t]].size())swap(a[2*t],a[2*t+1]);
  u64 l=c[a[0]].size()*b[a[1]].size(),r=c[a[2]].size()*b[a[3]].size();if(l+r<cost){cost=l+r;best=a;if(l>r)swap_ranges(best.begin(),best.begin()+2,best.begin()+2);}
 }
 auto a=best;u64 l=c[a[0]].size()*b[a[1]].size(),r=c[a[2]].size()*b[a[3]].size();z.work0=l;z.work1=r;if(l+r>cap||l>2500000){z.status=-1;return z;}
 struct Hit{u32 i,j;int rot;};unordered_map<Key,Hit,Hash> m;m.reserve(l);
 for(u32 i=0;i<c[a[0]].size();i++)for(u32 j=0;j<b[a[1]].size();j++){auto [k,t]=canon(add(c[a[0]][i].first,b[a[1]][j].first));m.try_emplace(k,Hit{i,j,t});}
 for(u32 i=0;i<c[a[2]].size();i++)for(u32 j=0;j<b[a[3]].size();j++){
  auto [k,t]=canon(sub(target,add(c[a[2]][i].first,b[a[3]][j].first)));auto it=m.find(k);if(it==m.end())continue;
  auto h=it->second;z.status=1;z.rows[a[0]]=rotate_row(c[a[0]][h.i].second,h.rot);z.rows[a[1]]=rotate_row(b[a[1]][h.j].second,h.rot);z.rows[a[2]]=rotate_row(c[a[2]][i].second,t);z.rows[a[3]]=rotate_row(b[a[3]][j].second,t);Key s{};for(auto row:z.rows)s=add(s,sig(row));assert(s==target);return z;
 }return z;
}

int main(){for(unsigned a=0;a<12;a++)for(unsigned b=0;b<12;b++){assert(add(rep(a),rep(b))==rep((a+b)%12));assert(sub(rep(a),rep(b))==rep((a+12-b)%12));}
 for(unsigned x=1;x<100;x++){Row r{};for(int j=0;j<K;j++){unsigned d=((x*2654435761u)>>(2*(j%12)))%4;r[d]|=(1u<<j)|(j?1u<<(Q-j):0);}Key a=sig(r);for(int k=0;k<11;k++){assert(a==sig(rotate_row(r,k)));a=rotate_key(a);}} int id;Profile p;while(cin>>id){for(auto& [u,v]:p)cin>>u>>v;Key t{(rep(10)&~31ULL)|9,11}; if(id<0){ // invariant positive target from rows with constant nonzero positions
  t={};for(auto [u,v]:p){u64 count;auto b=bank(u,v,count);auto it=find_if(b.begin(),b.end(),[](auto a){return rotate_key(a.first)==a.first;});assert(it!=b.end());t=add(t,it->first);}}

 auto z=solve(p,t);cout<<id<<","<<z.status<<","<<z.work0<<","<<z.work1;for(auto x:z.counts)cout<<","<<x;for(auto x:z.sizes)cout<<","<<x;for(auto r:z.rows)for(auto x:r)cout<<","<<x;cout<<endl;}
}
