#define HAD16_LIBRARY
#include "lift_affine.cpp"
#include <random>
// Independent scalar elimination over F3, with ordinary integer entries.
using Scalar=array<int,24>;
int mod3(int v){return (v%3+3)%3;}
int scalar_rank(vector<Scalar> a){int r=0;for(int j=0;j<24&&r<int(a.size());j++){int k=r;while(k<int(a.size())&&!a[k][j])++k;if(k==int(a.size()))continue;swap(a[k],a[r]);int d=a[r][j];for(int z=j;z<24;z++)a[r][z]=mod3(d*a[r][z]);for(int i=r+1;i<int(a.size());i++){int f=a[i][j];for(int z=j;z<24;z++)a[i][z]=mod3(a[i][z]-f*a[r][z]);}++r;}return r;}
bool scalar_possible(const array<vector<Group>,4>&b,V target){vector<Scalar> columns;for(auto &g:b){const V &v0=g[0].vals[0];for(auto&v:g)for(auto&w:v.vals){Scalar col{};for(int j=0;j<24;j++){int d=w[j]-v0[j];assert(d%3==0);col[j]=mod3(d/3);}columns.push_back(col);}target=subv(target,v0);}Scalar t{};for(int j=0;j<24;j++){if(target[j]%3)return false;t[j]=mod3(target[j]/3);}int r=scalar_rank(columns);columns.push_back(t);return scalar_rank(columns)==r;}
int main(){mt19937 rng(87601);W n=0,positive=0,negative=0,rejected=0;for(int k=0;k<3000;k++){
 array<vector<Group>,4>b;array<LiftSpan,4>s;Key target{};V itarget{};
 for(int f=0;f<4;f++){W u=0,v=0;for(int j=0;j<12;j++){u|=W(rng()%4)<<(4*j);v|=W(rng()%4)<<(4*j);}V base{};for(auto &x:base)x=rng()%3;int n=1+rng()%4;for(int i=0;i<n;i++){W x=u,y=v;for(int j=0;j<12;j++){x|=W(rng()%2)<<(4*j+2);y|=W(rng()%2)<<(4*j+2);}V z=base;for(auto &q:z)q+=3*(int(rng()%9)-4);b[f].push_back({{x,y,z[12]},{z},highbits(x,y)});}auto&g=b[f][rng()%n];target=add(target,g.key);itarget=addv(itarget,g.vals[0]);s[f]=make_lift_span(b[f]);}
 if(k%2)itarget[1]+=k%4==1?1:3;
 array<const vector<Group>*,4>a{&b[0],&b[1],&b[2],&b[3]};array<const LiftSpan*,4>sp{&s[0],&s[1],&s[2],&s[3]};bool pass=lift_possible(sp,itarget);assert(pass==scalar_possible(b,itarget));W hits=0;for(auto &w:b[0])for(auto &x:b[1])for(auto&y:b[2])for(auto&z:b[3])if(add(add(w.key,x.key),add(y.key,z.key))==target&&addv(addv(w.vals[0],x.vals[0]),addv(y.vals[0],z.vals[0]))==itarget)++hits;
 assert(!hits||pass);Counters old,cur;join(a,target,itarget,100000,old);join(a,target,itarget,100000,cur,nullptr,&sp);assert(old.hits==cur.hits&&cur.hits==hits);assert(!old.skipped&&!cur.skipped);assert(cur.central_bound==cur.central_ops+cur.avoided);assert((k%2)||hits);assert(!(k%2)||!hits);positive+=bool(hits);negative+=!hits;rejected+=!pass;n+=7;}
 cout<<"controls=3000 positive="<<positive<<" negative="<<negative<<" rejected="<<rejected<<" assertions="<<n<<'\n';
}
