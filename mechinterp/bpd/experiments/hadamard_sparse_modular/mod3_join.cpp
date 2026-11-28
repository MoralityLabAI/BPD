// Exact F3 group-ring projection of compressed half rows. Cap is per job;
// every capped case remains unresolved. Two trit bit planes, no fingerprints.
#include <array>
#include <vector>
#include <map>
#include <algorithm>
#include <unordered_set>
#include <iostream>
#include <fstream>
#include <cassert>
#include <cstdint>
#include <bit>
#include <cmath>
using namespace std;
using W=uint64_t;using I=uint32_t;
constexpr W M=(1ULL<<24)-1, CM=((1ULL<<12)-1)<<12;
constexpr I Q=(1U<<23)-1;
constexpr int B=432;
W add(W a,W b){W a1=a&M,a2=a>>24,b1=b&M,b2=b>>24,a0=~(a1|a2)&M,b0=~(b1|b2)&M;return ((a0&b1)|(a1&b0)|(a2&b2))|(((a0&b2)|(a2&b0)|(a1&b1))<<24);}
W neg(W a){return ((a&M)<<24)|(a>>24);}W sub(W a,W b){return add(a,neg(b));}
W trit(int n,int j){n=(n%3+3)%3;return n?1ULL<<(j+24*(n-1)):0;}
I shift(I x,int t){return ((x<<t)|(x>>(23-t)))&Q;}
struct Row{I p,n;array<I,12> pp,nn;W s;};
Row row(int s,I p,I n){if(s==1)p|=1;if(s==2)n|=1;assert(!(p&n));Row r{p,n,{},{},0};for(int t=0;t<12;t++){r.pp[t]=shift(p,t);r.nn[t]=shift(n,t);int a=popcount(p&r.pp[t])+popcount(n&r.nn[t])-popcount(p&r.nn[t])-popcount(n&r.pp[t]);r.s|=trit(a,t);}return r;}
map<int,vector<Row>> rows;
void gen(int s,int p,int n,int pos,I a,I b,vector<Row>&out){if(pos==12){if(!p&&!n)out.push_back(row(s,a,b));return;}int rem=12-pos;I m=(1U<<pos)|(1U<<(23-pos));if(p+n<rem)gen(s,p,n,pos+1,a,b,out);if(p)gen(s,p-1,n,pos+1,a|m,b,out);if(n)gen(s,p,n-1,pos+1,a,b|m,out);}
const vector<Row>& getrow(int i){auto it=rows.find(i);if(it!=rows.end())return it->second;vector<Row> r;gen(i/144,i/12%12,i%12,1,0,0,r);assert(!r.empty());return rows.emplace(i,move(r)).first->second;}
W cross(const Row&a,const Row&b){W s=0;for(int t=0;t<12;t++){int n=popcount(a.p&b.pp[t])+popcount(a.n&b.nn[t])-popcount(a.p&b.nn[t])-popcount(a.n&b.pp[t]);s|=trit(n,t+12);}return s;}
struct Bank{vector<W> c,j;W expanded=0;};map<int,Bank> banks;
const Bank& getbank(int i){auto it=banks.find(i);if(it!=banks.end())return it->second;auto &a=getrow(i/B),&b=getrow(i%B);Bank out;out.expanded=W(a.size())*b.size();unordered_set<W> j,c;for(auto&r:a)for(auto&s:b){W z=cross(r,s);j.insert(add(r.s,s.s)|z);c.insert(z);}out.c.assign(c.begin(),c.end());out.j.assign(j.begin(),j.end());sort(out.c.begin(),out.c.end());sort(out.j.begin(),out.j.end());return banks.emplace(i,move(out)).first->second;}
// Canonicalize the first bank under the common shift-class multiplier rotation.
// Using reps for any one summand preserves existence of solutions to invariant targets.
W rotate(W a,int k){W z=0;for(int t=0;t<12;t++){int i=min(t*k%23,23-t*k%23);for(int s:{0,12,24,36})if(a>>(t+s)&1)z|=1ULL<<(i+s);}return z;}
vector<W> reps(const vector<W>& a){vector<W> r;for(W x:a){bool keep=true;for(int k=2;k<=11;k++){if(rotate(x,k)<x){keep=false;break;}}if(keep)r.push_back(x);}return r;}
map<pair<int,int>,vector<W>> repcache;
const vector<W>& getrep(int id,int mode){auto key=make_pair(id,mode);auto it=repcache.find(key);if(it!=repcache.end())return it->second;auto &b=getbank(id);return repcache.emplace(key,reps(mode?b.j:b.c)).first->second;}
struct Job{int id;W rough;array<int,4>b;W mult;array<W,4> count;};
struct Plan{W cost=~0ULL;array<const vector<W>*,4>a{};};
Plan plan(const Job&j,int mode){Plan p;array<const vector<W>*,4>b;for(int i=0;i<4;i++){auto&k=getbank(j.b[i]);b[i]=mode?&k.j:&k.c;}for(int a=0;a<4;a++)for(int c=0;c<4;c++)if(c!=a){int d=-1,e=-1;for(int i=0;i<4;i++)if(i!=a&&i!=c){if(d<0)d=i;else e=i;}auto& r=getrep(j.b[a],mode);W x=W(r.size())*b[c]->size(),y=W(b[d]->size())*b[e]->size();if(x+y<p.cost)p={x+y,x<=y?array<const vector<W>*,4>{&r,b[c],b[d],b[e]}:array<const vector<W>*,4>{b[d],b[e],&r,b[c]}};}return p;}
// Counts both insertion and search ops. Positive early stop is allowed.
pair<bool,W> solve(const Plan&p,W target){auto&a=p.a;unordered_set<W> tab;tab.reserve(a[0]->size()*a[1]->size());W work=0;for(W x:*a[0])for(W y:*a[1]){tab.insert(add(x,y));++work;}for(W x:*a[2])for(W y:*a[3]){++work;if(tab.contains(sub(target,add(x,y))))return {true,work};}assert(work==p.cost);return {false,work};}
int main(int argc,char**argv){assert(argc==7);ifstream in(argv[1]);int mode=stoi(argv[2]),start=stoi(argv[3]),end=stoi(argv[4]);W cap=stoull(argv[5]);ofstream out(argv[6]);vector<Job> js;Job j;while(in>>j.id>>j.rough){for(int&k:j.b)in>>k;in>>j.mult;for(auto&x:j.count)in>>x;js.push_back(j);}W target=0;for(int t=0;t<12;t++){target|=trit(1,t+12);if(mode)target|=trit(t?2:1,t);}
 W ops=0,n=0,pos=0,neg=0,capped=0,weighted[3]{};for(int i=start;i<min(end,int(js.size()));i++){auto&j=js[i];auto p=plan(j,mode);for(int k=0;k<4;k++)assert(getbank(j.b[k]).expanded==j.count[k]);if(p.cost>cap){++capped;weighted[2]+=j.mult;out<<j.id<<",CAP,"<<j.mult<<','<<p.cost<<",0\n";continue;}auto[h,w]=solve(p,target);out<<j.id<<','<<(h?"POS":"NEG")<<','<<j.mult<<','<<p.cost<<','<<w<<'\n';ops+=w;++n;if(h){++pos;weighted[0]+=j.mult;}else{++neg;weighted[1]+=j.mult;}}
 cerr<<"mode="<<mode<<" range="<<start<<":"<<end<<" pos="<<pos<<" neg="<<neg<<" cap="<<capped<<" weighted="<<weighted[0]<<","<<weighted[1]<<","<<weighted[2]<<" ops="<<ops<<" bank_types="<<banks.size()<<'\n';
}
