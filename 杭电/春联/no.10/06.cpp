#include <bits/stdc++.h>
#define int long long
#define ull unsigned long long
#define i128 __int128
#define ld long double
#define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
#define pb push_back
#define eb emplace_back
#define pii pair<int, int>
#define vc vector
#define vi vector<int>
#define vvi vector<vi>
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define all1(x) x.begin() + 1, x.end()
#define INF 0x7fffffffffffffff
#define inf 0x7fffffff
// #define endl endl << flush
#define endl '\n'
#define WA AC
#define TLE AC
#define MLE AC
#define RE AC
#define CE AC
using namespace std;
const string AC = "Accepted";
const int MOD = 1e9 + 7;
const int mod = 998244353;
int n, m, x;
bool vis[100005 * 6];
int dis[100005 * 6];
int a[100005 * 6];
struct edge
{
	int to, w, next;
} e[1000005 * 4];
int head[100005 * 6], tot = 0;
void add(int u, int v, int w)
{
	e[++tot].to = v;
	e[tot].w = w;
	e[tot].next = head[u];
	head[u] = tot;
}
void Murasame()
{
	cin >> n >> m >> x;
	tot = 0;
	ff(i, 1, 6 * n) head[i] = 0;
	ff(i, 1, n) cin >> a[i];
	ff(i, 1, m)
	{
		int u, v, w;
		cin >> u >> v >> w;
		ff(j, 0, 2)
		{
			add(u + j * n, v + ((j + 1) % 3) * n, w);
		}
	}
	ff(i, 1, 6 * n)
	{
		dis[i] = INF;
		vis[i] = false;
	}
	dis[1] = 0;
	ff(i, 1, n)
	{
		ff(j, 0, 2)
		{
			add(i + j * n, a[i] + 3 * n + ((j + 1) % 3) * n, x);
			add(a[i] + 3 * n + ((j + 1) % 3) * n, i + ((j + 1) % 3) * n, 0);
		}
	}
	auto cmp = [](pii a, pii b){return a.se > b.se;};
	priority_queue<pii, vector<pii>, decltype(cmp)> pq(cmp);
	pq.push({1, 0});
	while (!pq.empty())
	{
		int u = pq.top().fi;
		pq.pop();
		if (vis[u])
			continue;
		vis[u] = true;
		for (int i = head[u]; i; i = e[i].next)
		{
			int v = e[i].to;
			if (dis[v] > dis[u] + e[i].w)
			{
				dis[v] = dis[u] + e[i].w;
				pq.push({v, dis[v]});
			}
		}
	}
	cout << (dis[n] >= INF ? -1 : dis[n]) << endl;
}
signed main()
{
	ios::sync_with_stdio(false);
	cin.tie(0);
	cout.tie(0);
	int _T = 1;
	//
	cin >> _T;
	while (_T--)
	{
		Murasame();
	}
	return 0;
}