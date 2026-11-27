// Exact fixed-A D syndrome + integer S autocorrelation join.
// All finite searches cap by upper-bound pair work supplied by independently enumerated row counts.
#include <array>
#include <vector>
#include <map>
#include <set>
#include <unordered_map>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <cassert>
#include <bit>
#include <chrono>
using namespace std;
using W=uint64_t;using WW=unsigned __int128;using I=uint32_t;
constexpr W MASK=(1ULL<<24)-1;
struct XY{int u,v;};
constexpr XY ax[4]={{1,0},{-1,0},{0,1},{0,-1}};
constexpr XY ds[9]={{0,0},{1,1},{1,-1},{-1,1},{-1,-1},{2,0},{-2,0},{0,2},{0,-2}};
W add(W a,W b){W a1=a&MASK,a2=a>>24,b1=b&MASK,b2=b>>24,a0=~(a1|a2)&MASK,b0=~(b1|b2)&MASK;return ((a0&b1)|(a1&b0)|(a2&b2))|(((a0&b2)|(a2&b0)|(a1&b1))<<24);}
W neg(W a){return ((a&MASK)<<24)|(a>>24);}W sub(W a,W b){return add(a,neg(b));}
W trit(int n,int j){n=(n%3+3)%3;return n?1ULL<<(j+24*(n-1)):0;}
int idx(int x){x=(x%23+23)%23;return min(x,23-x);}
int costkey(int u,int v){return 64*u+v;}
struct Pat{W mass,code,mult;};
using PBank=map<int,vector<Pat>>;
vector<PBank> patterns;
vector<map<int,W>> counts;
struct Row{W sig,mass,code;};
using Bank=map<int,vector<Row>>;
struct Central{array<int,12> a;array<vector<int>,4> pos;W basis[12][9]{};
 Central(I code){for(int i=0;i<12;i++){a[i]=(code>>(2*i))&3;if(i)pos[a[i]].push_back(i);}for(int j=0;j<12;j++)for(int d=0;d<9;d++)for(int t=0;t<12;t++){
   XY l=ax[a[idx(t-j)]],r=ax[a[idx(t+j)]];XY b=j?XY{l.u+r.u,l.v+r.v}:l;
   basis[j][d]|=trit(b.u*ds[d].u,t)|trit(b.v*ds[d].v,12+t);
 }}
 void expand(const Pat&p,int g,int offset,W sig,W code,vector<Row>&out)const{
  if(g==4){assert(offset==12);out.push_back({sig,p.mass,code});return;}
  int n=pos[g].size();array<int,11>d{};for(int j=0;j<n;j++)d[j]=(p.code>>(4*(offset+j)))&15;
  do{W s=sig,c=code;for(int j=0;j<n;j++){s=add(s,basis[pos[g][j]][d[j]]);c|=W(d[j])<<(4*pos[g][j]);}expand(p,g+1,offset+n,s,c,out);}while(next_permutation(d.begin(),d.begin()+n));
 }
 Bank make(int id,const set<int>&valid,W &expanded)const{Bank b;for(int k:valid){auto it=patterns[id].find(k);assert(it!=patterns[id].end());auto &out=b[k];for(auto&p:it->second){int d=p.code&15;size_t n=out.size();expand(p,0,1,basis[0][d],d,out);assert(out.size()-n==p.mult);}expanded+=out.size();}return b;}
};
WW sum_mass(W a,W b){WW z=0;while(b){int t=countr_zero(b);b&=b-1;z|=WW(a)<<t;}return z;}
bool hasmass(W a,W b,W c,W d,int m){WW x=sum_mass(a,b),y=sum_mass(c,d);for(int i=max(0,m-92);i<=min(92,m);i++)if((x>>i&1)&&(y>>(m-i)&1))return true;return false;}
W target(const array<Central,4>&a,int tu,int tv){W r=0;for(int t=0;t<12;t++){int u=(t==0?573:-26),v=(t==0?599:0);for(auto &row:a)for(int j=0;j<23;j++){auto x=ax[row.a[idx(j)]],y=ax[row.a[idx(j+t)]];u-=x.u*y.u;v-=x.v*y.v;}assert(u%12==0&&v%12==0);if(!t)assert(u/12==tu&&v/12==tv);r|=trit(u/12,t)|trit(v/12,12+t);}return r;}
bool full_s(const array<Central,4>&a,const array<W,4>&d){XY S[4][23];for(int f=0;f<4;f++)for(int j=0;j<23;j++){XY x=ax[a[f].a[idx(j)]],y=ds[(d[f]>>(4*idx(j)))&15];S[f][j]={x.u+6*y.u,x.v+6*y.v};}for(int t=0;t<12;t++){int u=0,v=0;for(int f=0;f<4;f++)for(int j=0;j<23;j++){u+=S[f][j].u*S[f][(j+t)%23].u;v+=S[f][j].v*S[f][(j+t)%23].v;}if(u!=(t? -26:573)||v!=(t?0:599))return false;}return true;}
struct Pair{const Row*a,*b;};
struct Job{int h,w,f,j,tu,tv,tm;W bound;array<int,4>o;array<I,4>a;array<int,4>ids;};
struct Group{int u,v;W n=0;vector<pair<int,int>>bins;};
map<int,Group> groups(const map<int,W>&a,const map<int,W>&b,int tu,int tv){map<int,Group>g;for(auto [i,x]:a)for(auto [j,y]:b){int u=i/64+j/64,v=i%64+j%64;if(u>tu||v>tv)continue;auto &z=g[costkey(u,v)];z.u=u;z.v=v;z.n+=x*y;z.bins.push_back({i,j});}return g;}
void solve(const Job&j,ostream&out,ostream&hits,W cap){if(j.bound>cap){out<<j.h<<' '<<j.w<<" CAP 0 0 0 0 0\n";return;}array<Central,4>a={Central(j.a[0]),Central(j.a[1]),Central(j.a[2]),Central(j.a[3])};W R=target(a,j.tu,j.tv);auto o=j.o;
 auto g0=groups(counts[j.ids[o[0]]],counts[j.ids[o[1]]],j.tu,j.tv),g1=groups(counts[j.ids[o[2]]],counts[j.ids[o[3]]],j.tu,j.tv);
 // counts[] include a union of costs over all witness classes. Complement filtering here
 // may admit extra costs compared to ranked per-class bound; restrict using exact job total.
 set<int> valid[4];vector<pair<int,int>>pairs;W bound=0;for(auto &[k,g]:g0){int l=costkey(j.tu-g.u,j.tv-g.v);auto it=g1.find(l);if(it==g1.end())continue;pairs.push_back({k,l});bound+=g.n+it->second.n;for(auto [x,y]:g.bins){valid[o[0]].insert(x);valid[o[1]].insert(y);}for(auto [x,y]:it->second.bins){valid[o[2]].insert(x);valid[o[3]].insert(y);}}
 if(bound>cap){out<<j.h<<' '<<j.w<<" CAP "<<bound<<" 0 0 0 0\n";return;}
 W expanded=0,work=0,synd=0,mass=0,solved=0;Bank b[4];for(int f=0;f<4;f++)b[f]=a[f].make(j.ids[f],valid[f],expanded);
 for(auto [k,l]:pairs){auto *g=&g0[k],*h=&g1[l];array<int,4>p=o;if(g->n>h->n){swap(g,h);p={o[2],o[3],o[0],o[1]};}
  unordered_multimap<W,Pair>tab;tab.reserve(g->n);for(auto [x,y]:g->bins)for(auto &r:b[p[0]][x])for(auto &s:b[p[1]][y]){tab.emplace(add(r.sig,s.sig),Pair{&r,&s});++work;}
  for(auto [x,y]:h->bins)for(auto &r:b[p[2]][x])for(auto &s:b[p[3]][y]){++work;auto range=tab.equal_range(sub(R,add(r.sig,s.sig)));for(auto it=range.first;it!=range.second;++it){++synd;auto &v=it->second;if(!hasmass(v.a->mass,v.b->mass,r.mass,s.mass,j.tm))continue;++mass;array<W,4>d;d[p[0]]=v.a->code;d[p[1]]=v.b->code;d[p[2]]=r.code;d[p[3]]=s.code;if(full_s(a,d)){++solved;hits<<j.h<<' '<<j.w;for(W z:d)hits<<' '<<z;hits<<'\n';}}}
 }
 assert(work==bound);out<<j.h<<' '<<j.w<<" COMPLETE "<<work<<' '<<expanded<<' '<<synd<<' '<<mass<<' '<<solved<<'\n';
}
int main(int argc,char**argv){assert(argc==5);ifstream pat(argv[1]),jobs(argv[2]);ofstream hits(argv[3]);W cap=stoull(argv[4]);int id,u,v;Pat p;while(pat>>id>>u>>v>>p.mass>>p.code>>p.mult){if(id>=int(patterns.size())){patterns.resize(id+1);counts.resize(id+1);}patterns[id][costkey(u,v)].push_back(p);counts[id][costkey(u,v)]+=p.mult;}
 Job j;int n=0;while(jobs>>j.h>>j.w>>j.f>>j.j>>j.tu>>j.tv>>j.tm>>j.bound){for(auto&v:j.o)jobs>>v;for(auto&v:j.a)jobs>>v;for(auto&v:j.ids)jobs>>v;solve(j,cout,hits,cap);if(++n%1000==0)cerr<<n<<" jobs\n";} }
