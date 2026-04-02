#include <bits/stdc++.h>
#define int long long
#define ull unsigned long long
#define i128 __int128
#define ld long double
#define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
#define pb push_back
#define eb emplace_back
#define vc vector
#define pii pair<int, int>
#define mpr make_pair
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define INF 0x7fffffffffffffff
#define inf 0x7fffffff
#define endl '\n'
#define WA AC
#define TLE AC
#define MLE AC
#define RE AC
#define CE AC
using namespace std;
const string AC = "Accepted";
int cnt;
void dfs(int x, vector<int> &a, vector<int> &vis)
{
    if (vis[x])
        return;
    vis[x] = 1;
    cnt++;
    dfs(a[x], a, vis);
}
void solve()
{
    int n;
    cin >> n;
    vector<int> a(n + 1), b(n + 1), vis(n + 1, 0);
    // vector<int> da(n + 1), db(n + 1);
    cnt = 0;
    for (int i = 1; i <= n; i++)
    {
        cin >> a[i];
        // da[a[i]] = i;
    }
    for (int i = 1; i <= n; i++)
    {
        cin >> b[i];
        // db[b[i]] = i;
        dfs(b[i], a, vis);
        cout << cnt << " ";
    }
    cout << endl;
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
        solve();
    }
    return 0;
}