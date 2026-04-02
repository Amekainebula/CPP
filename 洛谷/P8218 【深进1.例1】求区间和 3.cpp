#include <bits/stdc++.h>
#define int long long
#define i128 __int128
#define ld long double
#define ull unsigned long long
#define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
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
#define inf 0x7fffffff
#define endl '\n'
#define WA AC
#define TLE AC
#define MLE AC
#define RE AC
#define CE AC
using namespace std;
const string AC = "Accepted";
vector<int> tree(100005), a(100005);
void solve()
{
    int n, m;
    cin >> n;
    auto add = [&](int x, int v)
    {
        while (x <= n)
        {
            tree[x] += v;
            x += lowbit(x);
        }
    };
    auto sum = [&](int x)
    {
        int res = 0;
        while (x)
        {
            res += tree[x];
            x -= lowbit(x);
        }
        return res;
    };
    ff(i, 1, n)
    {
        cin >> a[i]; 
        add(i, a[i]);
    }
    cin >> m;
    while (m--)
    {
        int l, r;
        cin >> l >> r;
        cout << sum(r) - sum(l - 1) << endl;
    }
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int T = 1;
    // cin >> T;
    while (T--)
    {
        solve();
    }
    return 0;
}