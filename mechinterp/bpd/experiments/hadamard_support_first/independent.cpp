// Independent central-column audit: enumerate the support parametrization directly,
// use scalar autocorrelations and radix-3 keys, without the root table.
#include <array>
#include <vector>
#include <algorithm>
#include <bit>
#include <cstdint>
#include <cassert>
#include <iostream>
#include <unordered_map>
using namespace std;
constexpr int Q=23,K=12,N=4096,F=N-1,L=729;
using U=uint16_t;using I=uint32_t;using W=uint64_t;using Masks=array<U,4>;
array<int,N> mass{},canonical{},orbit_size{},sq{};
array<array<vector<I>,24>,N> bank;
vector<Masks> quads;
array<array<U,L>,L> plus6{},minus6{};
I add3(I a,I b){return plus6[a%L][b%L]+L*plus6[a/L][b/L];}
I minus3(I a,I b){return minus6[a%L][b%L]+L*minus6[a/L][b/L];}
int at(int m,int j){j=min(j,Q-j);return m>>j&1;}
int shift(int m,int a){int b=0;for(int i=0;i<K;i++)if(m>>i&1){int j=i*a%Q;j=min(j,Q-j);b|=1<<j;}return b;}
I scalar_signature(int p,int n){array<int,Q> x{};for(int i=0;i<Q;i++)x[i]=at(p,i)-at(n,i);I key=0,pow=1;for(int t=0;t<K;t++){int c=0;for(int i=0;i<Q;i++)c+=x[i]*x[(i+t)%Q];key+=pow*((c%3+3)%3);pow*=3;}return key;}
void init(){
 for(int a=0;a<L;a++)for(int b=0;b<L;b++){int x=a,y=b,p=1;for(int j=0;j<6;j++){plus6[a][b]+=p*((x%3+y%3)%3);minus6[a][b]+=p*((x%3+3-y%3)%3);x/=3;y/=3;p*=3;}}
 for(int p=0;p<N;p++){
  mass[p]=2*popcount(unsigned(p))-(p&1);sq[p]=shift(p,2);canonical[p]=p;for(int a=1;a<Q;a++)canonical[p]=min(canonical[p],shift(p,a));orbit_size[p]=(p==0||p==1||p==F||p==F-1)?1:11;
  for(int a=p;;a=(a-1)&p){int m=2*mass[a]-mass[p];if(m>=0)bank[p][m].push_back(scalar_signature(a,p^a));if(!a)break;}
  for(auto &b:bank[p]){sort(b.begin(),b.end());b.erase(unique(b.begin(),b.end()),b.end());}
 }
 for(int s=0;s<N;s++){
  // Independent scalar convolution with twelve basis vectors.
  array<int,K> col{};for(int k=0;k<K;k++)for(int t=0;t<K;t++){int j=(t-k+Q)%Q,l=(t+k)%Q;int v=at(s,j)^(k?at(s,l):0);col[k]|=v<<t;}
  array<int,N> prod{};for(int w=1;w<N;w++)prod[w]=prod[w&(w-1)]^col[countr_zero(unsigned(w))];assert(prod[s]==sq[s]);
  for(int w=0;w<N;w++){int t=1^s^w^sq[s]^sq[w]^prod[w];if(canonical[t]!=t)continue;Masks p={U(t),U(t^s),U(t^w),U(t^s^w^1)};if((mass[p[0]]+mass[p[1]]+mass[p[2]]+mass[p[3]])%12!=11)continue;quads.push_back(p);}
 }
 W total=0;for(auto p:quads)total+=orbit_size[p[0]];assert(total==5635620);cerr<<"canonical supports="<<quads.size()<<" weighted="<<total<<"\n";
}
using B=array<const vector<I>*,4>;using Profile=array<pair<int,int>,4>;
W cost(const B& b){W best=~0ULL;for(int j=1;j<4;j++){int k=j==1?2:1,l=6-j-k;best=min(best,W(b[0]->size())*b[j]->size()+W(b[k]->size())*b[l]->size());}return best;}
bool solve(const B& b,I target,W &work){array<int,4>a{};W best=~0ULL;for(int j=1;j<4;j++){int k=j==1?2:1,l=6-j-k;W x=W(b[0]->size())*b[j]->size(),y=W(b[k]->size())*b[l]->size();if(x+y<best){best=x+y;a=x<=y?array<int,4>{0,j,k,l}:array<int,4>{k,l,0,j};}}
 // Sorted vector, no dependence on hash table or packed-lane reduction.
 vector<I> left;left.reserve(b[a[0]]->size()*b[a[1]]->size());for(I x:*b[a[0]])for(I y:*b[a[1]])left.push_back(add3(x,y));work+=left.size();sort(left.begin(),left.end());
 for(I x:*b[a[2]])for(I y:*b[a[3]]){++work;if(binary_search(left.begin(),left.end(),minus3(target,add3(x,y))))return true;}return false;}
int main(){init();I tu=0,p=3;for(int i=1;i<K;i++){tu+=p;p*=3;}int id;W cap;Profile prof;while(cin>>id>>cap){for(auto &[u,v]:prof)cin>>u>>v;W total=0,work=0;bool hit=false;array<array<bool,N>,4> allowed{};for(int i=0;i<4;i++)for(int j=0;j<N;j++)allowed[i][j]=!bank[F^j][prof[i].first].empty()&&!bank[j][prof[i].second].empty();for(auto p:quads){bool ok=true;for(int i=0;i<4;i++)ok&=allowed[i][p[i]];if(!ok)continue;total+=orbit_size[p[0]];if(hit)continue;B bu,bv;for(int i=0;i<4;i++){bu[i]=&bank[F^p[i]][prof[i].first];bv[i]=&bank[p[i]][prof[i].second];}if(cost(bu)<=cost(bv)){if(solve(bu,tu,work)&&solve(bv,2,work))hit=true;}else if(solve(bv,2,work)&&solve(bu,tu,work))hit=true;}
 cout<<id<<","<<hit<<","<<total<<","<<work<<endl;}return 0;}
