// Full integer correlation necessary signature. A collision is a candidate, never an exclusion.
#define main old_central_main
#include "conditional_central.cpp"
#undef main
#include <random>
struct FP{W a,b;bool operator==(const FP&)const=default;bool operator<(const FP&x)const{return a==x.a?b<x.b:a<x.a;}};
FP operator+(FP a,FP b){return {a.a+b.a,a.b+b.b};}FP operator-(FP a,FP b){return {a.a-b.a,a.b-b.b};}FP operator*(FP a,int b){return {a.a*W(b),a.b*W(b)};}
struct FHash{size_t operator()(FP a)const{return a.a^rotl(a.b,31);}};
array<FP,12> sw,cw;FP sk[12][12],ck[12][12];
int idx(int x){x=(x%23+23)%23;return min(x,23-x);}
void initfp(){mt19937_64 rng(87401);for(auto &x:sw)x={rng(),rng()};for(auto &x:cw)x={rng(),rng()};for(int t=0;t<12;t++)for(int j=0;j<23;j++){int a=idx(j),b=idx(j+t);sk[a][b]=sk[a][b]+sw[t];ck[a][b]=ck[a][b]+cw[t];}}
FP scalar(const Row&x,const Row&y,bool cross){FP z{};auto&w=cross?cw:sw;for(int t=0;t<12;t++)for(int j=0;j<23;j++)z=z+w[t]*(x.a[idx(j)]*y.a[idx(j+t)]);return z;}
struct Prep{FP s;array<FP,12> c;};Prep prep(const Row&x){Prep p{};for(int i=0;i<12;i++){for(int j=0;j<12;j++){p.s=p.s+sk[i][j]*(x.a[i]*x.a[j]);p.c[i]=p.c[i]+ck[i][j]*x.a[j];}}return p;}
FP bilinear(const Row&x,const Prep&y){FP z{};for(int i=0;i<12;i++)z=z+y.c[i]*x.a[i];return z;}
using FB=array<vector<FP>,N>;
FB fullbank(int hx,int hy,int sx,int sy,W c,W&raw,W&checks){auto a=rows(hx,sx,c&F,(c>>12)&F),b=rows(hy,sy,(c>>24)&F,(c>>36)&F);vector<Prep> ap,bp;for(auto x:a)ap.push_back(prep(x));for(auto y:b)bp.push_back(prep(y));FB out;raw+=W(a.size())*b.size();for(unsigned i=0;i<a.size();i++)for(unsigned j=0;j<b.size();j++){auto x=a[i],y=b[j];FP z=ap[i].s+bp[j].s+bilinear(x,bp[j]);if((i*b.size()+j)%997==0){assert(z==scalar(x,x,false)+scalar(y,y,false)+scalar(x,y,true));++checks;}out[x.plus^y.plus].push_back(z);}for(auto&b:out){sort(b.begin(),b.end());b.erase(unique(b.begin(),b.end()),b.end());}return out;}
int main(int argc,char**argv){assert(argc==5);init();initfp();ifstream h(argv[1]),in(argv[2]),wit(argv[3]);W cap=stoull(argv[4]);int nh;h>>nh;for(int&x:val)h>>x;hist.resize(nh);for(auto&a:hist)for(int&x:a)h>>x;map<int,Job> jobs;int id;Job j;while(in>>id>>j.mult){for(int&x:j.h)in>>x;for(int&x:j.s)in>>x;jobs[id]=j;}FP target{};for(int t=0;t<12;t++)target=target+sw[t]*(t?-52:2344)+cw[t]*(-26);array<W,4> codes;while(wit>>id){for(W &x:codes)wit>>x;j=jobs.at(id);W raw=0,checks=0,tuples=0,work=0,keys=0,plans=0,hits=0,skipped=0;array<FB,4>b;for(int f=0;f<4;f++){b[f]=fullbank(j.h[2*f],j.h[2*f+1],j.s[2*f],j.s[2*f+1],codes[f],raw,checks);for(auto &v:b[f])keys+=v.size();}
 for(auto p:supports){array<const vector<FP>*,4>a;bool ok=true;for(int i=0;i<4;i++){a[i]=&b[i][p[i]];if(a[i]->empty()){ok=false;break;}}if(!ok)continue;++tuples;W best=~0ULL;array<int,4> o{};for(int i=1;i<4;i++){int k=i==1?2:1,l=6-i-k;W x=W(a[0]->size())*a[i]->size(),y=W(a[k]->size())*a[l]->size();if(x+y<best){best=x+y;o=x<=y?array<int,4>{0,i,k,l}:array<int,4>{k,l,0,i};}}plans+=best;if(work+best>cap){++skipped;continue;}unordered_set<FP,FHash> tab;for(auto x:*a[o[0]])for(auto y:*a[o[1]]){tab.insert(x+y);++work;}for(auto x:*a[o[2]])for(auto y:*a[o[3]]){++work;if(tab.contains(target-(x+y)))++hits;}}
 assert(work<=cap);if(!skipped)assert(work==plans);cout<<id<<','<<j.mult<<','<<(hits?"FP_HIT":skipped?"CAP":"NEG")<<','<<raw<<','<<keys<<','<<tuples<<','<<plans<<','<<work<<','<<checks<<','<<hits<<','<<skipped<<endl;}return 0;}
