#include <bits/stdc++.h>
#define int long long
using namespace std;
const int N = 2e3 + 10;
const int INF = 1e18;
struct edge
{
    int v, w;
};
vector<edge> e[N];
vector<int> dis(N), vis(N), cnt(N), val(N);

queue<int> q;
bool spfa(int n){
    fill(dis.begin(), dis.end(), INF);
    dis[0] = 0;
    vis[0] = 1;
    q.push(0);
    while (!q.empty()){
        int u = q.front();
        q.pop();
        vis[u] = 0;
        for (auto &it : e[u]){
            int v = it.v, w = it.w;
            if (dis[v] > dis[u] + w){
                dis[v] = dis[u] + w;
                cnt[v] = cnt[u] + 1; // 记录最短路经过的边数
                if (cnt[v] >= n + 1)
                    return false;
                if (!vis[v]){
                    q.push(v);
                    vis[v] = 1;
                }
            }
        }
    }
    return true;
}
vector<int> dis2(N);
void dijkstra(int s){
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> q;
    fill(dis2.begin(), dis2.end(), INF);
    fill(vis.begin(), vis.end(), 0);
    dis2[s] = 0;
    q.push({0, s});
    while (!q.empty()){
        int u = q.top().second;
        q.pop();
        if (vis[u])
            continue;
        vis[u] = 1;
        for (auto &it : e[u]){
            int v = it.v, w = it.w;
            if (dis2[v] > dis2[u] + w){
                dis2[v] = dis2[u] + w;
                q.push({dis2[v], v});
            }
        }
    }
}
void solve(){
    ios::sync_with_stdio(false);
    cin.tie(0);
    int n, m;
    cin >> n >> m;
    for (int i = 1; i <= n; i++){
        cin >> val[i];
    }
    for (int i = 1; i <= m; i++){
        int u, v, w;
        cin >> u >> v >> w;
        e[u].push_back({v, w});
    }
    for (int i = 1; i <= n; i++){
        e[0].push_back({i, 0});
    }
    if (!spfa(n)){
        cout << -1 << '\n';
        return;
    }
    for (int i = 1; i <= n; i++){
        for (auto &it : e[i]){
            it.w += dis[i] - dis[it.v];
        }
    }
    int ans = LLONG_MAX;
    for (int i = 1; i <= n; i++){
        dijkstra(i);
        int tmp = 0;
        for (int j = 1; j <= n; j++){
            if (dis2[j] == INF){
                tmp += 1e9 * val[j];
            }
            else{
                tmp += val[j] * (dis2[j] + dis[j] - dis[i]);
            }
        }
        ans = min(ans, tmp);
    }
    cout << ans << '\n';
}

signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int _T = 1;
    // cin>>_T;
    while (_T--)
    {
        solve();
    }
    return 0;
}