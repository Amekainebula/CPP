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

void solve()
{
    int n, q;
    cin >> n >> q;
    vc<int> arr(n + 1), tree(n + 1), a(n + 1);
    vc<int> ans;
    auto lowbit = [](int x)
    {
        return x & -x;
    };
    auto add = [&](int x, int y)
    {
        while (x <= n)
        {
            tree[x] += y;
            x += lowbit(x);
        }
    };
    auto sum = [&](int l, int r)
    {
        int res = 0;
        for (int i = l - 1; i; i -= lowbit(i))
            res -= tree[i];
        for (int i = r; i; i -= lowbit(i))
            res += tree[i];
        return res;
    };
    for (int i = 1; i <= n; i++)
    {
        cin >> a[i];
        // arr[i] = a[i] + arr[i - 1];
        add(i, a[i]);
    }
    int cnt = 0;
    while (q--)
    {
        int op;
        int x, y;
        cin >> op;
        cin >> x >> y;
        if (op == 1)
        {
            add(x, y - a[x]);
            a[x] = y;
        }
        else if (op == 2)
        {
            int xx = min(x, y), yy = max(x, y);
            cnt++;
            int tmp = sum(1, yy) / 100 - sum(1, xx - 1) / 100;
            //cout << sum(1, xx - 1) << ' ' << sum(1, yy) << ' ' << tmp << endl;
            // cout << tmp << endl;
            ans.pb(tmp * cnt);
        }
    }
    int res = -1;
    for (auto x : ans)
    {
        if (res == -1)
            res = x;
        else
            res ^= x;
    }
    cout << res << endl;
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