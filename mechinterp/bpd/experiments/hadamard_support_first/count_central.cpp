// Exact enumeration of all central sign completions, extending independent.cpp.
// Uses raw sign multiplicities. Counts first-row support-normalized tuples and
// labeled tuples by restoring the number of orientations of that support.
#define main independent_main
#include "independent.cpp"
#undef main
using Wide=unsigned __int128;
struct Count {I key;W n;};
array<array<vector<Count>,24>,N> multi;
void build_counts(){for(int p=0;p<N;p++){
 array<unordered_map<I,W>,24> tmp;
 for(int a=p;;a=(a-1)&p){int m=2*mass[a]-mass[p];if(m>=0)++tmp[m][scalar_signature(a,p^a)];if(!a)break;}
 for(int j=0;j<24;j++){for(auto [k,n]:tmp[j])multi[p][j].push_back({k,n});assert(multi[p][j].size()==bank[p][j].size());}
}}
using C=array<const vector<Count>*,4>;
W mc(const C& b){W best=~0ULL;for(int j=1;j<4;j++){int k=j==1?2:1,l=6-j-k;best=min(best,W(b[0]->size())*b[j]->size()+W(b[k]->size())*b[l]->size());}return best;}
W count_join(const C& b,I target,W &work){array<int,4>a{};W best=~0ULL;for(int j=1;j<4;j++){int k=j==1?2:1,l=6-j-k;W x=W(b[0]->size())*b[j]->size(),y=W(b[k]->size())*b[l]->size();if(x+y<best){best=x+y;a=x<=y?array<int,4>{0,j,k,l}:array<int,4>{k,l,0,j};}}
 unordered_map<I,W> left;left.reserve(b[a[0]]->size()*b[a[1]]->size());for(auto x:*b[a[0]])for(auto y:*b[a[1]]){++work;left[add3(x.key,y.key)]+=x.n*y.n;}
 W total=0;for(auto x:*b[a[2]])for(auto y:*b[a[3]]){++work;auto it=left.find(minus3(target,add3(x.key,y.key)));if(it!=left.end())total+=it->second*x.n*y.n;}return total;}
void print128(Wide x){if(x>=10)print128(x/10);cout<<unsigned(x%10);}
int main(){init();build_counts();I tu=0,p=3;for(int i=1;i<K;i++){tu+=p;p*=3;}int id;W cap;Profile prof;while(cin>>id>>cap){for(auto &[u,v]:prof)cin>>u>>v;W supports=0,work=0,compat=0;Wide tuples=0,labeled=0;array<array<bool,N>,4> allowed{};for(int i=0;i<4;i++)for(int j=0;j<N;j++)allowed[i][j]=!multi[F^j][prof[i].first].empty()&&!multi[j][prof[i].second].empty();for(auto p:quads){bool ok=true;for(int i=0;i<4;i++)ok&=allowed[i][p[i]];if(!ok)continue;++compat;C bu,bv;for(int i=0;i<4;i++){bu[i]=&multi[F^p[i]][prof[i].first];bv[i]=&multi[p[i]][prof[i].second];}W a,b;if(mc(bu)<=mc(bv)){a=count_join(bu,tu,work);if(!a)continue;b=count_join(bv,2,work);}else{b=count_join(bv,2,work);if(!b)continue;a=count_join(bu,tu,work);}if(!a||!b)continue;++supports;Wide c=Wide(a)*b;tuples+=c;labeled+=c*orbit_size[p[0]];}
 cout<<id<<","<<compat<<","<<supports<<","<<work<<",";print128(tuples);cout<<",";print128(labeled);cout<<endl;}}
