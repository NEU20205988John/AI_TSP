#include<bits/stdc++.h>
#include<cstdio>
#include<iostream>
#include<cstring>
#include<cstdlib>
#include<algorithm>
#include<queue>
#include<map>
#include<set>
#include<cmath>
#include<vector>
#include<cassert>
#include<ctime>
#include<functional>
using namespace std;
const int N=34;//34个城市34个点
const int M = 33*34/2;//C(34,2)条边=561,数组记得再加1
const int INF = 0x3f3f3f3f;
double ans_final=INF;

typedef long long ll;
using namespace std;
#define pii pair<int,int>
#define pll pair<ll,ll>
#define pil pair<int,ll>
#define pli pair<ll,int>
#define fi first
#define se second
#define mp make_pair
#define pb push_back

struct city{//城市结构体定义，名字+经纬度
    string name;
    double longtitude;
    double latitude;
}a[N+1];

struct edge{
    int u,v;
    double len;
    friend bool operator<(const edge & edge1,const edge & edge2){
        return edge1.len<edge2.len;
    }
}e[M+1];
vector<struct edge> MST;
vector<int> odd_degree_vertex;
int degree[N+1];

int p[N+1];//并查集用到的父亲节点,带路径压缩
int find(int x){return p[x]==x?x:p[x]=find(p[x]);}//并查集的查找函数，查找过程中进行路经压缩，减少复杂度

vector<int> linjie[N+1];
vector<int> visit_order;
bool vis[N+1];

double Kruskal(){
    memset(degree,0,sizeof(degree));//度数清零
    double ans=0;//没有卵用，最小生成树代价这里用不到，待会计算奇度定点
    for(int i=1;i<=N;i++){
        p[i]=i;//将N个城市划分为N个连通分量（并查集中每个人是自己的父亲）
    }
    sort(e+1,e+M+1);//对建的M条边按照边权进行排序,O(nlogn)
    for(int i=1;i<=M;i++){
        int x=find(e[i].u);//找到x所在集合代表元素
        int y=find(e[i].v);//找到y所在集合代表元素
        if(x!=y){
            ans+=e[i].len;
            p[x]=y;
            //MST.push_back({x,y,e[i].len});这么写是错误的
            MST.push_back({e[i].u,e[i].v,e[i].len});
            degree[e[i].u]++;
            degree[e[i].v]++;
        }
    }
    return ans;
}

double calc(int i,int j){
    return sqrt(pow(a[i].longtitude-a[j].longtitude,2)+pow(a[i].latitude-a[j].latitude,2));
}

double cost(vector<int> vc){
    double ans=0;
    for(int i=1;i<vc.size();i++){
        ans+=calc(vc[i-1],vc[i]);
    }
    ans+=calc(vc[0],vc[vc.size()-1]);
    return ans;
}

void init_data(){
    ifstream inFile("city.csv",ios::in);
    string lineStr;
    int i=1;
    while (getline(inFile, lineStr)) {
        stringstream ss(lineStr);//此处使用stringstream，可以将已经读进来的一行string变为流，便于后续用分隔符读取流中的信息。
        string str;
        vector<string> lineArray;
        while (getline(ss, str, ',')){
            lineArray.push_back(str);
        }
        a[i].name=lineArray[0];
        a[i].longtitude = stod(lineArray[1]);
        a[i].latitude = stod(lineArray[2]);
        i++;
    }
    int cnt=1;
    for(int i=1;i<=34;i++){
        for(int j=i+1;j<=34;j++){
            e[cnt++]={i,j,calc(i,j)};//建边
        }
    }
}


int match[N+1];
//int match[N+1] = {0,7,0,12,0,10,0,1,0,0,5,14,3,0,11,0,0,0,29,26,0,0,0,0,0,0,19,0,0,18,0,0,33,32,0};
//19 0 18 0 32 0 26 0 0 29 12 11 0 33 0 0 0 3 1 0 0 0 0 0 0 7 0 0 10 0 0 5 14 0
// int match[N+1] = {0,19,0,18,0,32,0,26,0,0,29,12,11,0,33,0,0,0,3,1,0,0,0,0,0,0,7,0,0,10,0,0,5,14,0};
int G[N+1][N+1];
vector<int> Euler;
vector<int>Hamilton;

struct Bi{
    int x1,x2;
    friend bool operator<(const Bi &a1, const Bi &a2){
        return a1.x1<a2.x1;
    }
    friend bool operator>=(const Bi &a1, const Bi &a2){
        return a1.x1>=a2.x1;
    }
};
int vis2[N+1][N+1];

void euler(int u){
    Euler.push_back(u);
    for(int v=1;v<=N;v++){
        if(G[u][v]!=vis2[u][v]&&G[v][u]!=vis2[v][u]){
            //cout<<u<<"->"<<v<<endl;
            vis2[u][v]++;
            vis2[v][u]++;
            euler(v);
        }
    }
}

vector<int> final_path;

vector<int> _my_opt(vector<vector<int> > solutions){
    vector<int> Tbest=solutions[0];
    vector<int> Tpie=Tbest;
    vector<int> T;
    final_path = Tbest;
    cout<<"正在进行my_opt优化!"<<endl;
    for(int k=0;k<solutions.size();k++){
        T=solutions[k];
        bool nochange=true;
        cout<<"第"<<k+1<<"次"<<endl;
        do{
            Tbest=T;
            vector<int>::iterator it;
            vector<int>::iterator it_2;
            for(it=Tpie.begin()+1;it!=Tpie.end()-2;it++){
                Tpie=solutions[k];//it's a copy，局部最优解的一个副本
                //枚举需要提出来的元素的下标
                int element = *it;//暂存元素
                Tpie.erase(it);//将该元素删除（提出来，稍后再插入）
                vector<int> Tpie2=Tpie;//每次保存一个Tpie的副本
                for(it_2=it+1;it_2!=Tpie.end();it_2++){
                    Tpie.insert(it_2,element);//插入回去
                    if(cost(Tpie)<cost(Tbest)){
                        Tbest=Tpie;
                        nochange=false;
                        cout<<cost(Tpie)<<endl;
                    }
                    Tpie=Tpie2;
                }
            }
            T=Tbest;
        }while(nochange);
        if(cost(Tbest)<cost(final_path)){
            final_path=Tbest;
        }
    }
    return Tbest;
}
vector<int> _2opt(vector<vector<int> > solutions){
    vector<int> Tbest=solutions[0];
    vector<int> Tpie=Tbest;
    vector<int> T;
    final_path = Tbest;
    cout<<"正在进行2-opt优化!"<<endl;
    for(int k=0;k<solutions.size();k++){
        T=solutions[k];
        bool nochange=true;
        cout<<"第"<<k+1<<"次"<<endl;
        do{
            Tbest=T;
            for(int i=1;i<=31;i++){
                for(int j=i+1;j<=32;j++){
                    //Tpie=solutions[k];
                    swap(Tpie[i],Tpie[j]);
                    if(cost(Tpie)<cost(Tbest)){
                        Tbest=Tpie;
                        nochange=false;
                        cout<<cost(Tpie)<<endl;
                    }else{
                        swap(Tpie[i],Tpie[j]);
                    }
                    
                }
            }
            T=Tbest;
            //这个优化方法是：2opt，遍历所有优化的可能（交换两个数）
            //////////////////////////////
        }while(nochange);
        if(cost(Tbest)<cost(final_path)){
            final_path=Tbest;
        }
    }
    return Tbest;
}

namespace my_flower_tree{
 #define N1 34*2+1
	struct edge1{
		int u,v;ll w;
		edge1(){}
		edge1(int u,int v,ll w):u(u),v(v),w(w){}
	};
	// Graph
	int n,n_x; // [1, n]: point; [n+1, n_x]: flower
	edge1 g[N1][N1]; // adjacent matrix
	// flower
	vector<int>flower[N1]; // nodes in flower i (outer flower)
	int root[N1]; // flower root, root<=n root=i: normal nodes
	int flower_from[N1][N1]; // flower_from[b][x]: outermost flower in b that contains x
	// slack
	ll label[N1]; // node label, [1, n] point label, [n+1, n_x] flower label
	int col[N1]; // color saved at flower root
	int slv[N1]; // slack node of NON LEAF NODES, slv[y]=x z(x,y) min_x
	// match
	int mat[N1]; // match, mat[x]=y (x,y)\in E
	int fa[N1]; // fa in cross tree
	int vis[N1]; // if in path
	
	queue<int>Q; // bfs queue
	
	// calculate slv
	inline ll calc_slv(edge1 e){return label[e.u]+label[e.v]-e.w;}
	inline void update_slv(int u,int v){if(!slv[v]||calc_slv(g[u][v])<calc_slv(g[slv[v]][v]))slv[v]=u;}
	inline void recalc_slv(int u){
		slv[u]=0;
		for(int i=1;i<=n;i++)if(g[i][u].w>0&&root[i]!=u&&col[root[i]]==1)update_slv(i,u);
	}
	
	// only push nodes, not flowers
	void q_push(int x){
		if(x<=n)Q.push(x);//如果是普通节点，加入Q队列中，否则找x对应的花，递归调用q_push，也就是把花里面所含元素加入队列Q
		else for(auto p:flower[x])q_push(p);
	}
	
	// set root of all nodes in x to r
	void set_root(int x,int r){
		root[x]=r;
		if(x>n)for(auto p:flower[x])set_root(p,r);
	}
	
	// return a (+-)^k path in flower b from root[b] to x
    //获取某朵花b中，从花托到x的一条交错路径：对花连出去边的两种情况分别处理，返回一个指针。
	int get_even_path_in_flower(int b,int x){
		int pr=find(flower[b].begin(),flower[b].end(),x)-flower[b].begin();
        //在这朵花里找到真实的x值在哪
		assert(b>n&&b<=n_x&&pr<flower[b].size()); // b is flower, x in b
		if(pr%2==0)return pr;//如果路径长度为偶数，则直接返回pr
		reverse(flower[b].begin()+1,flower[b].end());
        //因为根据花的定义，花其实是个奇环，将花托之外的数字反转，得到的一定是另一个方向的路径
		return flower[b].size()-pr;//因此可以保证一定能返回一个偶数的路径长度
	}
	
	// set (u,v) match, can be flower
    /*
        某次增广使得需要设置u匹配v，u可能是花（v不管):对于u为花的情况，从花托到真实u 
    的边都进行翻转匹配，最后花托移位，旋转花到正确位置。
    */
	void set_match(int u,int v){
		mat[u]=g[u][v].v;//真实两点进行匹配
		if(u>n){//如果是花的花，花托到真实u的点的边一个个拎出来，翻转匹配
			edge1 e=g[u][v];
			int xr=flower_from[u][e.u];//flower_from[B][i] 表示最大的包含i的B的子花
            //cout<<"u = "<<u<<",e.u = "<<e.u<<endl;
            //那么此处xr得到的是u这朵花内最大的包含u的子花
            //注意u是花的编号(>34),而e.u是u这个节点的实际编号
			int pr=get_even_path_in_flower(u,xr);
			for(int i=0;i<pr;i++)set_match(flower[u][i],flower[u][i^1]);
            //那么对于我得到的路径，上面的每一个点都需要翻转匹配
			set_match(xr,v);
			rotate(flower[u].begin(),flower[u].begin()+pr,flower[u].end()); // change receptacle
		}
	}
	
	// link 2 S points
    /*
    两个点到根的路径，加上两点之间的边，能构成一条增广路。
    */
	void side_augment(int u,int v){
		int nv=root[mat[u]],nu=root[fa[nv]];
		while(1){
			set_match(u,v);
			u=nu,v=nv;
			if(!nv)break;
			set_match(nv,nu);
			nv=root[mat[u]],nu=root[fa[nv]];
		}
	}
	void linkSS(int u,int v){
		side_augment(u,v); 
		side_augment(v,u);
	}
	
	int get_lca(int u,int v){
		static int t=0;
		++t; // to avoid clearing vis
		while(u||v){
			if(vis[u]==t)return u;
			vis[u]=t;
			u=root[mat[u]];
			if(u)u=root[fa[u]];
			if(!u)swap(u,v);
		}
		return 0;
	}
	
	void add_blossom(int u,int v,int r){
		int i,b=n+1;
		while(b<=n_x&&root[b])b++;
		if(b>n_x)++n_x;
		// clear
		col[b]=1;label[b]=0;mat[b]=mat[r];flower[b].clear();
		for(i=1;i<=n_x;i++)g[i][b].w=g[b][i].w=0;
		for(i=1;i<=n;i++)flower_from[b][i]=0;
		// construct flower
		while(u!=r){
			flower[b].pb(u);u=root[mat[u]];q_push(u);
			flower[b].pb(u);u=root[fa[u]];
		}
		flower[b].pb(r);
		reverse(flower[b].begin(),flower[b].end());
		while(v!=r){
			flower[b].pb(v);v=root[mat[v]];q_push(v);
			flower[b].pb(v);v=root[fa[v]];
		}
		// set as outermost flower
		set_root(b,b);
		// calculate slack
		for(auto p:flower[b]){
			for(i=1;i<=n_x;i++){
				// set to min slave
				if(!g[b][i].w||calc_slv(g[p][i])<calc_slv(g[b][i])){
					g[b][i]=g[p][i];
					g[i][b]=g[i][p];
				}
			}
			for(i=1;i<=n;i++)if(flower_from[p][i])flower_from[b][i]=p;
		}
		recalc_slv(b);
	}
	
	// only expand outermost blossom b, b is T(white) blossom
	void expand_blossom(int b){
		int i,x;
		for(auto p:flower[b])set_root(p,p);
		x=flower_from[b][g[b][fa[b]].u];
		// [0,pr]: (+-)^k, insert into tree, add black to queue
		int pr=get_even_path_in_flower(b,x);
		col[x]=2;fa[x]=fa[b];
		for(i=0;i<pr;i+=2){
			// from bottom to upper layer in tree
			int white=flower[b][i];
			int black=flower[b][i+1];
			col[black]=1;col[white]=2;
			fa[white]=g[black][white].u;
			slv[black]=slv[white]=0;
			q_push(black);
		}
		// others: color=0
		for(i=pr+1;i<flower[b].size();i++){
			col[flower[b][i]]=0;
			recalc_slv(flower[b][i]);
		}
		// delete b
		root[b]=0;
		flower[b].clear();
	}
	
	// found_edge
    /*
    尝试增广一条等边：如果对点未染过色，则必然已经有匹配，将其匹配染色后丢入队列；
                如果对点是黑点，分两种情况，如果 LCA=0即不在同一花内，则 linkSS，否则添加花。
    */
	int augment_path(edge1 e){
		int u=root[e.u],v=root[e.v];
		if(!col[v]){
			assert(mat[v]);
			fa[v]=e.u;//匹配就是把v的父亲标记为u，本来就连过边了
			col[v]=2;//标记为已匹配
			int nu=root[mat[v]];
			slv[nu]=slv[v]=0;
			col[nu]=1;//染成黑色
			q_push(nu);
		}else if(col[v]==1){//对于黑点来说，不在同一花内，则需要连接两个黑点linkSS，否则加花
			int r=get_lca(u,v);
			if(r)add_blossom(u,v,r);
			else return linkSS(u,v),1;
		}
		return 0;
	}
    
	
	int augment(){
		int i;
		memset(col,0,sizeof(int)*(n_x+1));//col，颜色
		memset(slv,0,sizeof(int)*(n_x+1));//z
		memset(fa,0,sizeof(int)*(n_x+1));//father，记录父亲节点
		Q=queue<int>();//bfs序列
		for(i=1;i<=n_x;i++)
			if(root[i]==i&&!mat[i]){
				// add all unmatched points
                //将所有未匹配的节点加入队列
				col[i]=1;//颜色标记为1
				q_push(i);
			}
		if(Q.empty())return 0;//bfs结束，return0，增广结束！
		while(1){
			while(!Q.empty()){
				int p=Q.front();Q.pop();//取得队列首部元素
				assert(col[root[p]]==1);//检查合法性
				for(i=1;i<=n;i++){
					if(g[p][i].w==0||root[i]==root[p])continue;//若没有建边或root相同则跳过
					// not in same flower
					ll d=calc_slv(g[p][i]);//计算所需代价z值
					if(!d){if(augment_path(g[p][i]))return 1;}//如果代价为0，则计算augment_path，尝试增广一条等边
					else if(col[root[i]]!=2)update_slv(p,root[i]);//计算出松弛的大小，更新两个z值
                    //
				}
			}
			ll delta=INF;//要让delta尽可能小，先赋值为INF
			// calc delta
			for(i=1;i<=n;i++)if(col[root[i]]==1)delta=min(delta,label[i]);//遍历所有点，未匹配点的代价中找到最小值
			for(i=n+1;i<=n_x;i++)if(root[i]==i&&col[i]==2)delta=min(delta,label[i]/2);//匹配点中代价为连边的权值，记得除以2
			for(i=1;i<=n_x;i++){
				if(root[i]!=i||!slv[i])continue;
				if(!col[i])delta=min(delta,calc_slv(g[slv[i]][i]));
				else if(col[i]==1)delta=min(delta,calc_slv(g[slv[i]][i])/2);
			}
			// update label
			for(i=1;i<=n;i++){
				if(col[root[i]]==1)label[i]-=delta;
				else if(col[root[i]]==2)label[i]+=delta;
			}
            //对于普通的点，白色点加上期望值，黑色点减去期望值即可
            //但是对于花里面的点，会受到影响，因此黑花的点再加上2delta，白的点再减去2delta
            //最终效果就是所有的普通点，白色都减去了1个delta，黑色加上了一个delta，而对于花这个染黑了的整体，全部增加了1delta
			for(i=n+1;i<=n_x;i++){
				if(root[i]!=i)continue;
				if(col[i]==1)label[i]+=2*delta;
				else if(col[i]==2)label[i]-=2*delta;
			}
			for(i=1;i<=n;i++)if(label[i]<=0)return 0;
			for(i=1;i<=n_x;i++){
				if(root[i]!=i||!slv[i]||root[slv[i]]==i)continue;
				if(calc_slv(g[slv[i]][i])==0&&augment_path(g[slv[i]][i]))return 1;
			}
			// expand
			for(i=n+1;i<=n_x;i++)
				if(root[i]==i&&col[i]==2&&label[i]==0)
					expand_blossom(i);
		}
		return 0;
	}
	
	void init(int _n,vector<pair<ll,pii> >edges){
		int i,j;
		n=n_x=_n;//总共n个点
		memset(mat,0,sizeof(mat));//初始化匹配数组match，0表示没有匹配
		for(i=0;i<=n;i++){
			root[i]=i;//初始化每个节点的根为自己
			flower[i].clear();//每朵花都清空
			for(j=0;j<=n;j++){
				flower_from[i][j]=(i==j)?i:0;//flower_from[B][i] 表示最大的包含i的B的子花
				g[i][j]=edge1(i,j,0);//清空边权
			}
		}
		ll w_max=0;//记录最大边权
		for(auto pr:edges){//遍历所有边
			int u=pr.se.fi,v=pr.se.se;//获取边两端的两个顶点号
			ll w=pr.fi;//获取边权
			g[u][v]=edge1(u,v,w*2);//因为z数组要用到z/2，因此这里所有边权乘以2
			g[v][u]=edge1(v,u,w*2);
			w_max=max(w_max,w);
		}
		for(i=1;i<=n;i++)label[i]=w_max;//根据KM算法，二分图匹配里面每个点所需的最大贡献值，label也就是对偶问题里的z
	}
	
	pair<int,ll>calc(){
		int i,cnt=0;ll s=0;
		while(augment())++cnt;
        //不断尝试进行增广，记录增广次数
		for(i=1;i<=n;i++)if(mat[i]>i)s+=g[i][mat[i]].w/2;
        //对于每个点，检查是否成功匹配，且匹配的对方编号大于自己（那么对于另一个人肯定对方编号小于自己）匹配一次即可
        //sum每次加上的是匹配权值的一半（因为初始化的时候乘以二了）
		return mp(cnt,s);
	}
}


int main(){
    clock_t start,finish;
    double totaltime;
    start=clock();

    init_data();//读取csv文件并建图
    Kruskal();//Kruskal求最小生成树

    for(int i=1;i<=N;i++){
        if(degree[i]&1){
            odd_degree_vertex.push_back(i);//查找奇度顶点
        }
    }
    vector<pair<ll,pii> >edges;
    for(int i=0;i<odd_degree_vertex.size();i++){
        for(int j=i+1;j<odd_degree_vertex.size();j++){
            long long weight = (ll)((ll)INF-(double)100*(calc(odd_degree_vertex[i],odd_degree_vertex[j])));
            edges.pb(mp(weight,mp(odd_degree_vertex[i],odd_degree_vertex[j])));
            //将所有奇度顶点构成的完全图的边放入edges，便于带权带花树找 一般图最小权值匹配 这里需要注意，因为只能处理
            //只能处理正数权值，因此用一个很大的数字INF减去原本的边权，得到的是仍是一个正数，但可以通过找最大权值匹配而得到实际上的最小权值匹配。
            //*100是将其放大同等倍数，double小数点的值也放大。
            //cout<<odd_degree_vertex[i]<<" "<<odd_degree_vertex[j]<<" "<<(double)(INF-(double)100*(calc(odd_degree_vertex[i],odd_degree_vertex[j])))<<endl;
        }
    }

    my_flower_tree::init(N,edges);
    my_flower_tree::calc();
    for(int i=1;i<=N;i++){
        match[i]=my_flower_tree::mat[i];
    }

    for(int i=1;i<=34;i++){
        if(match[i]){
            MST.push_back({i,match[i],calc(i,match[i])});
            match[i]=0;
        }
        //这里说一下，i号点对应的匹配点为match[i]，因此将i和match[i]作为一条边的两个顶点进行建边
        //为避免重复建边，i号点建完match[i]就可以清空了
    }
    //求得最小匹配后，进行加边，奇度顶点进行匹配，然后试图求欧拉回路。

    //建立邻接矩阵
    for(int i=0;i<MST.size();i++){
        int x=MST[i].u;
        int y=MST[i].v;
        G[x][y]++;
        G[y][x]++;//双向建边，用的时候用一个拆另一个
    }
    vector<vector<int> > solutions;
    vector<vector<int> > new_solutions;
    
    for(int i=1;i<=N;i++){
        Euler.clear();
        Hamilton.clear();
        memset(vis,false,sizeof(vis));
        euler(i);//尝试以每个点为起点，找欧拉回路
        for(int i=0;i<Euler.size();i++){
            if(!vis[Euler[i]]){
                //cout<<Euler[i]-1<<endl；
                Hamilton.push_back(Euler[i]);
                //建立汉密尔顿回路
            }
            vis[Euler[i]]=true;
        }
        //寻找欧拉回路，从一个点出发，走一条边拆掉双向的边（入度--）
        
        double ans=0;
        for(int i=1;i<Hamilton.size();i++){
            ans+=calc(Hamilton[i-1],Hamilton[i]);
        }
        ans+=calc(Hamilton[0],Hamilton[Hamilton.size()-1]);
        //cout<<"ans = "<<ans<<endl;
        ans_final=min(ans_final,ans);
        if(ans<163.50){
            //只保留较为优秀的结果作为后续优化的输入
            vector<int> temp = Hamilton;
            solutions.push_back(temp);
        }
        for(int i=0;i<MST.size();i++){
            int x=MST[i].u;
            int y=MST[i].v;
            G[x][y]++;
            G[y][x]++;//双向建边，用的时候用一个拆另一个
        }
    }
    cout<<"Christofides算法初步结果为:"<<ans_final<<endl;
    new_solutions=solutions;
    //cout<<"solutions.size = "<<solutions.size()<<endl;
    vector<int> solution = _2opt(solutions);//进行2opt优化
    cout<<"经过2opt优化后的最小代价为:"<<cost(solution)<<endl;


    solutions.clear();
    solutions.push_back(solution);//对最终路径再进行一遍优化，变成循环链表那种，34种可能全部遍历
    for(int i=0;i<solutions.size();i++){
        vector<int> temp=solutions[i];
        do{
            int element=temp[0];
            temp.erase(temp.begin());
            temp.push_back(element);
            new_solutions.push_back(temp);
        }while(temp!=solutions[i]);
    }
    //cout<<"new_solutions.size = "<<new_solutions.size()<<endl;
    solution = _my_opt(solutions);

    //cout<<"cost of solution = "<<cost(solution)<<endl;
    //vector<int> solution = _2opt(new_solutions);
    cout<<"经过自行研究的优化方法优化后，最小代价为："<<cost(solution)<<endl;
    // new_solutions.clear();
    // new_solutions.push_back(final_path);
    // solution = _my_opt(new_solutions);
    cout<<"cost of final_path = "<<cost(final_path)<<endl;
    cout<<"最终路径为："<<endl;
    for(int i=0;i<final_path.size();i++){
        if(i){
            cout<<"->";
        }
        cout<<a[final_path[i]].name;
    }
    cout<<endl;

    cout<<"最终TSP路径为:"<<endl;
    for(int i=0;i<final_path.size();i++){
        if(i)cout<<",";
        cout<<final_path[i]-1;
    }
    cout<<endl;

    finish=clock();
    totaltime=(double)(finish-start) / CLOCKS_PER_SEC;
    cout<<"此程序的运行时间为"<<totaltime<<"秒！"<<endl;

    return 0;
}
//12,7,14,13,9,4,8,5,6,0,1,3,10,16,15,17,20,28,26,23,19,24,29,31,32,33,30,25,22,18,27,21,2,11
//33,30,25,22,18,27,21,2,11,12,7,14,13,9,4,8,5,6,0,1,3,10,16,15,20,17,28,26,23,19,24,29,32,31