// Central mod8 and exact dot first; complete 24-coordinate integer join second.
#define main previous_main
#if __has_include("../HAD14/conditional_central.cpp")
#include "../HAD14/conditional_central.cpp"
#else
#include "../hadamard_conditional_integer/conditional_central.cpp"
#endif
#undef main
using V=array<int16_t,24>;
V addv(V a,const V&b){for(int j=0;j<24;j++)a[j]+=b[j];return a;}
V subv(V a,const V&b){for(int j=0;j<24;j++)a[j]-=b[j];return a;}
struct VH{size_t operator()(const V&a)const{size_t h=0xcbf29ce484222325ULL;for(auto x:a)h=(h^uint16_t(x))*0x100000001b3ULL;return h;}};

#include <unordered_map>
#include <chrono>
uint32_t highbits(W u,W v){uint32_t z=0;for(int j=0;j<12;j++){z|=((u>>(4*j+2))&1)<<j;z|=((v>>(4*j+2))&1)<<(j+12);}return z;}
struct Group{Key key;vector<V> vals;uint32_t high=0;};
struct Span{uint32_t offset=0;array<uint32_t,24> pivot{};vector<uint32_t> basis;int lo=0,hi=0;};
void insert_basis(array<uint32_t,24>&p,uint32_t x){while(x){int i=31-countl_zero(x);if(!p[i]){p[i]=x;return;}x^=p[i];}}
bool contains_basis(const array<uint32_t,24>&p,uint32_t x){while(x){int i=31-countl_zero(x);if(!p[i])return false;x^=p[i];}return true;}
Span make_span(const vector<Group>&g){assert(!g.empty());Span s;s.offset=g.front().high;s.lo=s.hi=g.front().key.d;for(auto&x:g){insert_basis(s.pivot,x.high^s.offset);s.lo=min(s.lo,x.key.d);s.hi=max(s.hi,x.key.d);}for(auto x:s.pivot)if(x)s.basis.push_back(x);return s;}

using GB=array<vector<Group>,N>;
struct Entry{Key key;V value;};
GB groupedbank(int hx,int hy,int sx,int sy,W c,W&raw,W&unique,W&groups){
 auto a=rows(hx,sx,c&F,(c>>12)&F),b=rows(hy,sy,(c>>24)&F,(c>>36)&F);
 array<vector<Entry>,N> tmp;raw+=W(a.size())*b.size();
 for(auto&x:a)for(auto&y:b){V z{};for(int t=0;t<12;t++)for(int j=0;j<23;j++){int k=min(j,23-j),l=(j+t)%23;l=min(l,23-l);z[t]+=x.a[k]*x.a[l]+y.a[k]*y.a[l];z[t+12]+=x.a[k]*y.a[l];}
 int p=x.plus,q=y.plus,v=p^q;Key k{sig(p&q,F^(p|q)),sig(p&~q,q&~p),z[12]};tmp[v].push_back({k,z});}
 GB out;for(int i=0;i<N;i++){auto&t=tmp[i];sort(t.begin(),t.end(),[](const Entry&a,const Entry&b){return a.key==b.key?a.value<b.value:a.key<b.key;});
 for(auto&e:t){if(out[i].empty()||!(out[i].back().key==e.key)){out[i].push_back({e.key,{},highbits(e.key.u,e.key.v)});++groups;}auto&v=out[i].back().vals;if(v.empty()||v.back()!=e.value){v.push_back(e.value);++unique;}}}for(auto&b:out)if(!b.empty())for(auto&g:b){assert((g.key.u&0x333333333333ULL)==(b.front().key.u&0x333333333333ULL));assert((g.key.v&0x333333333333ULL)==(b.front().key.v&0x333333333333ULL));}return out;
}
struct Counters{W tuples=0,central_ops=0,compatible=0,integer_ops=0,hits=0,skipped=0,central_bound=0,span_reject=0,dot_reject=0,avoided=0;};
void join(const array<const vector<Group>*,4>&a,Key target,const V&itarget,W cap,Counters&c,const array<const Span*,4>* spans=nullptr){
 array<int,4> o{};W best=~0ULL;for(int i=1;i<4;i++){int k=i==1?2:1,l=6-i-k;W x=W(a[0]->size())*a[i]->size(),y=W(a[k]->size())*a[l]->size();if(x+y<best){best=x+y;o=x<=y?array<int,4>{0,i,k,l}:array<int,4>{k,l,0,i};}}
 c.central_bound+=best;
 // For fixed V support, the mod4 low bits of both central PAFs are fixed.
 // Sum the low lanes (each <=3, so no carry between 4-bit lanes). The high bits
 // then add by XOR after subtracting the known bit-2 carry of the low sum.
 constexpr W LO=0x333333333333ULL;
 W lu=0,lv=0;for(auto g:a){auto k=g->front().key;lu+=k.u&LO;lv+=k.v&LO;}
 assert((lu&LO)==(target.u&LO)&&(lv&LO)==(target.v&LO));
 uint32_t need=highbits(lu^target.u,lv^target.v);
 if(spans){int lo=0,hi=0;uint32_t x=need;array<uint32_t,24> basis{};for(auto s:*spans){lo+=s->lo;hi+=s->hi;x^=s->offset;}if(target.d<lo||target.d>hi){++c.dot_reject;c.avoided+=best;return;}
 for(auto s:*spans)for(auto v:s->basis)insert_basis(basis,v);if(!contains_basis(basis,x)){++c.span_reject;c.avoided+=best;return;}}
 if(c.central_ops+c.integer_ops+best>cap){++c.skipped;return;}c.central_ops+=best;
 auto pack=[](uint32_t hi,int dot){return (W(uint32_t(dot))<<24)|hi;};
 using GP=pair<const Group*,const Group*>;
 // Flat chained hash table, reused across tuples. No fingerprinting of central keys.
 struct Slot{W key;int head;};static vector<Slot> slots;static vector<GP> pairvec;static vector<int> next;
 size_t n=a[o[0]]->size()*a[o[1]]->size(),sz=1;while(sz<2*n)sz*=2;slots.assign(sz,{0,-1});pairvec.clear();next.clear();pairvec.reserve(n);next.reserve(n);
 auto hash=[&](W key){key^=key>>30;key*=0xbf58476d1ce4e5b9ULL;key^=key>>27;key*=0x94d049bb133111ebULL;return (key^(key>>31))&(sz-1);};
 auto slot=[&](W key){size_t i=hash(key);while(slots[i].head>=0&&slots[i].key!=key)i=(i+1)&(sz-1);return i;};
 for(auto&x:*a[o[0]])for(auto&y:*a[o[1]]){W key=pack(x.high^y.high,x.key.d+y.key.d);auto i=slot(key);next.push_back(slots[i].head);slots[i]={key,int(pairvec.size())};pairvec.emplace_back(&x,&y);}
 for(auto&x:*a[o[2]])for(auto&y:*a[o[3]]){W key=pack(need^x.high^y.high,target.d-x.key.d-y.key.d);auto i=slot(key);for(int j=slots[i].head;j>=0;j=next[j]){++c.compatible;auto&u=*pairvec[j].first;auto&v=*pairvec[j].second;assert(add(add(u.key,v.key),add(x.key,y.key))==target);
 array<const vector<V>*,4>b{&u.vals,&v.vals,&x.vals,&y.vals};array<int,4> p{};W cost=~0ULL;for(int i=1;i<4;i++){int k=i==1?2:1,l=6-i-k;W l1=W(b[0]->size())*b[i]->size(),l2=W(b[k]->size())*b[l]->size();if(l1+l2<cost){cost=l1+l2;p=l1<=l2?array<int,4>{0,i,k,l}:array<int,4>{k,l,0,i};}}
 if(c.central_ops+c.integer_ops+cost>cap){++c.skipped;continue;}
 unordered_set<V,VH> s;s.reserve(b[p[0]]->size()*b[p[1]]->size());for(auto&z:*b[p[0]])for(auto&t:*b[p[1]]){s.insert(addv(z,t));++c.integer_ops;}for(auto&z:*b[p[2]])for(auto&t:*b[p[3]]){if(s.contains(subv(itarget,addv(z,t))))++c.hits;++c.integer_ops;}
 }}
}

#ifndef HAD15_LIBRARY
int main(int argc,char**argv){assert(argc==5);init();ifstream h(argv[1]),in(argv[2]),wit(argv[3]);W cap=stoull(argv[4]);int nh;h>>nh;for(int&x:val)h>>x;hist.resize(nh);for(auto&a:hist)for(int&x:a)h>>x;map<int,Job>jobs;int id;Job j;while(in>>id>>j.mult){for(int&x:j.h)in>>x;for(int&x:j.s)in>>x;jobs[id]=j;}
 Key target{0,0,-26};V itarget{};itarget[0]=2344;for(int t=0;t<12;t++){target.u|=W((t?-26:573)&7)<<(4*t);target.v|=W((t?0:599)&7)<<(4*t);if(t)itarget[t]=-52;itarget[t+12]=-26;}
 // Synthetic positive and cap controls: same central and integer signatures, known sum.
 {array<vector<Group>,4>b;Key k{};V v{};for(int f=0;f<4;f++){Key a{W(f),W(f+1),f-2};V w{};w[f]=f+1;b[f].push_back({a,{w},highbits(a.u,a.v)});k=add(k,a);v=addv(v,w);}array<const vector<Group>*,4>a{&b[0],&b[1],&b[2],&b[3]};Counters c;join(a,k,v,100,c);assert(c.hits==1&&c.compatible==1&&!c.skipped);Counters d;join(a,k,v,0,d);assert(d.skipped==1&&!d.hits);}
 array<W,4>codes;while(wit>>id){for(W&x:codes)wit>>x;j=jobs.at(id);W raw=0,unique=0,groups=0;Counters c;array<GB,4>b;for(int f=0;f<4;f++)b[f]=groupedbank(j.h[2*f],j.h[2*f+1],j.s[2*f],j.s[2*f+1],codes[f],raw,unique,groups);
 array<array<Span,N>,4> spans;for(int f=0;f<4;f++)for(int i=0;i<N;i++)if(!b[f][i].empty())spans[f][i]=make_span(b[f][i]);
 for(auto p:supports){array<const vector<Group>*,4>a;bool ok=true;for(int i=0;i<4;i++){a[i]=&b[i][p[i]];if(a[i]->empty()){ok=false;break;}}if(!ok)continue;++c.tuples;array<const Span*,4> ss{&spans[0][p[0]],&spans[1][p[1]],&spans[2][p[2]],&spans[3][p[3]]};join(a,target,itarget,cap,c,&ss);}
 cout<<id<<','<<j.mult<<','<<(c.hits?"HIT":c.skipped?"CAP":"NEG")<<','<<raw<<','<<unique<<','<<groups<<','<<c.tuples<<','<<c.central_bound<<','<<c.central_ops<<','<<c.compatible<<','<<c.integer_ops<<','<<c.hits<<','<<c.skipped<<','<<c.span_reject<<','<<c.dot_reject<<','<<c.avoided<<endl;
 }return 0;}

#endif
