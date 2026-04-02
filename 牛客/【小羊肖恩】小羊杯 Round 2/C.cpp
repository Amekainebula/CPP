#include <bits/stdc++.h>
#define int long long
#define i128 __int128
#define ld long double
#define ull unsigned long long
#define lowbit(x) (x & -x)
#define pb push_back
#define eb emplace_back
#define pii pair<int, int>
#define mpr make_pair
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define rall(x) x.rbegin(), x.rend()
#define sz(x) (int)(x).size()
#define INF 0x7fffffffffffffff
#define endl '\n' 
using namespace std;
void dfs(int u,int cnt,map<int,int>&mp,vector<int>&t,vector<bool>&vis,vector<vector<int>>&g)
{
    mp[cnt]++;
    //cout<<cnt<<" "<<u<<endl;
    vis[u]=true;
    
    for(auto v:g[u])
    {
        if(vis[v]) continue;
        else dfs(v,cnt+1,mp,t,vis,g);
    }
}
void solve()
{
    int n;
    cin >> n;
    vector<vector<int>>g(n+5);
    
    vector<int>t(n+5,0);
    
    for(int i=1;i<n;i++)
    {
        int u,v;
        cin >> u >> v;
        g[u].pb(v);
        g[v].pb(u);
        t[u]++;
        t[v]++;
    } 
    // auto dfs=[&](this auto &&dfs,int u,int cnt)->void
    // {
    //     vis[u]=true;
    //     if(g[u].size()==1)
    //     {
    //         a.pb(cnt);
    //         return;
    //     }
    //     for(auto v:g[u])
    //     {
    //         //temp++;
    //         if(vis[v]) continue;
    //         else dfs(v,cnt+1);
    //     }
    //     //if(temp==1) a.pb(cnt);
    // };
   bool flag=true;
    for(int i=1;i<=n;i++)
    {
        map<int,int>mp;
        if(t[i]==1)
        {
            vector<bool>vis(n+5,false);
            dfs(i,1,mp,t,vis,g);
            int ok=0;
            flag=true;
            for(auto x:mp)
            {
                if(x.se>1)
                {
                    if(ok==0)
                    ok=1;
                    else
                    {
                        flag=false;
                        break;
                    }

                }
                if(x.se==1)
                ok=0;
            }
            if(flag)break;
        }
    }
    flag==true?cout<<"YES\n":cout<<"NO\n";
    // dfs(4,1,a,t,vis,g);
    //sort(all(a));
    
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int T = 1;
    cin >> T;
    while (T--)
    {
        solve();
    }
    return 0;
}
