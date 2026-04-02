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
const int N = 1e5 + 6;
class node
{
public:
    int u, v, w;
};
int n, k;
int ans = 0;
map<pii, int> mp;
bool cmp(node a, node b)
{
    return a.w > b.w;
}
vi fa(N + 1);
int find(int x)
{
    return x == fa[x] ? x : fa[x] = find(fa[x]);
}
void Murasame()
{
    cin >> n >> k;
    vi a(n + 1);
    vi vis(n + 1);
    vc<node> g(n);
    ff(i, 1, n)
    {
        fa[i] = i;
    }
    ff(i, 1, k)
    {
        int x;
        cin >> x;
        a[x] = 1;
    }
    ff(i, 1, n - 1)
    {
        cin >> g[i].u >> g[i].v >> g[i].w;
        ans += g[i].w;
        // g2[y].pb({x,z});
    }
    sort(g.begin() + 1, g.begin() + n, cmp);
    ff(i, 1, n - 1)
    {
        auto [u, v, w] = g[i];
        int fx = find(u), fy = find(v);
        if (a[fx] && a[fy])
            continue;
        fa[fx] = fy;
        ans -= w;
        if (a[fx])
            a[fy] = 1;
        else if (a[fy])
            a[fx] = 1;
    }
    cout << ans << endl;
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int _T = 1;
    // cin >> _T;
    while (_T--)
    {
        Murasame();
    }
    return 0;
}