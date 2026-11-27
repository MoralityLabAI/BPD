#define HAD15_LIBRARY
#include "affine_join.cpp"
#include <random>
int main(){
 mt19937 rng(87501);W assertions=0,positive=0,negative=0,span_reject=0,dot_reject=0;
 for(int k=0;k<2000;k++){
  array<vector<Group>,4>b;Key target{};V itarget{};array<Span,4>s;
  for(int f=0;f<4;f++){
   W u=0,v=0;for(int t=0;t<12;t++){u|=W(rng()%4)<<(4*t);v|=W(rng()%4)<<(4*t);}
   int n=1+rng()%4;for(int i=0;i<n;i++){W x=u,y=v;for(int t=0;t<12;t++){x|=W(rng()%2)<<(4*t+2);y|=W(rng()%2)<<(4*t+2);}int dot=int(rng()%21)-10;V z{};for(auto &w:z)w=int(rng()%9)-4;z[12]=dot;b[f].push_back({{x,y,dot},{z},highbits(x,y)});}
   auto &g=b[f][rng()%n];target=add(target,g.key);itarget=addv(itarget,g.vals.front());s[f]=make_span(b[f]);
  }
  // A known positive on even cases; alter a high bit, dot, or integer coordinate otherwise.
  if(k%2){switch(k%3){case 0:target.u^=W(4)<<(4*(rng()%12));break;case 1:target.d+=40;break;default:itarget[0]+=40;}}
  W compatible=0,hits=0;for(auto &a:b[0])for(auto&c:b[1])for(auto&d:b[2])for(auto&e:b[3])if(add(add(a.key,c.key),add(d.key,e.key))==target){++compatible;if(addv(addv(a.vals[0],c.vals[0]),addv(d.vals[0],e.vals[0]))==itarget)++hits;}
  array<const vector<Group>*,4>a{&b[0],&b[1],&b[2],&b[3]};array<const Span*,4>sp{&s[0],&s[1],&s[2],&s[3]};Counters c,d;join(a,target,itarget,100000,c,&sp);join(a,target,itarget,100000,d);assert(!c.skipped&&!d.skipped);assert(c.compatible==compatible&&d.compatible==compatible);assert(c.hits==hits&&d.hits==hits);assert(c.central_bound==c.central_ops+c.avoided);assert((k%2)||hits);assert(!(k%2)||!hits);assert(c.central_ops+c.integer_ops<=100000);assertions+=7;positive+=bool(hits);negative+=!hits;span_reject+=c.span_reject;dot_reject+=c.dot_reject;
 }
 cout<<"positive="<<positive<<" negative="<<negative<<" assertions="<<assertions<<" span_reject="<<span_reject<<" dot_reject="<<dot_reject<<'\n';
}
