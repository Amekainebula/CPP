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
int n, k;
int dp[5000 + 1][5000 + 1];
int val[5000 + 1], cost[5000 + 1];
vi g[5000 + 1];
void dfs(int u)
{
    for (auto v : g[u])
    {
        ff(i, 0, k)
        {
            if (i + 1 <= k)
                dp[v][i + 1] = max(dp[v][i + 1], dp[u][i]);
            if (i + cost[v] <= k)
                dp[v][i + cost[v]] = max(dp[v][i + cost[v]], dp[u][i] + val[v]);
        }
        dfs(v);
        ff(i, 0, k) dp[u][i] = max(dp[u][i], dp[v][i]);
    }
}
void Murasame()
{
    cin >> n >> k;
    ff(i, 1, n) ff(j, 0, k) dp[i][j] = -INF; // i为节点，j为当前花费，dp[i][j]为最大收益
    ff(i, 2, n)
    {
        int x;
        cin >> x;
        g[x].pb(i);
    }
    // vi val(n + 1), cost(n + 1);
    ff(i, 1, n) cin >> val[i];
    ff(i, 1, n) cin >> cost[i];
    dp[1][0] = 0;
    dp[1][1] = 0;
    dp[1][cost[1]] = val[1];
    dfs(1);
    int ans = -INF;
    ff(i, 0, k) ans = max(ans, dp[1][i]);
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