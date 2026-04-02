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
    int n;
    cin >> n;
    vc<int> a(n + 1), b(n + 1);
    vc<int> da(n + 1), db(n + 1);
    vc<pii> ans;
    ff(i, 1, n) cin >> a[i], da[a[i]] = i;
    ff(i, 1, n) cin >> b[i], db[b[i]] = i;
    int cnt = 0;
    ff(i, 1, n)
    {
        if (a[i] != b[n - i + 1])
            break;
        if (i == n)
        {
            cout << 0 << endl;
            return;
        }
    }
    if (cnt >= 2)
    {
        cout << -1 << endl;
        return;
    }
    bool ok = 1;
    ff(i, 1, n)
    {
        if (a[i] == b[i] && ok)
        {
            if (n % 2)
            {
                swap(a[i], a[(n + 1) / 2]);
                swap(b[i], b[(n + 1) / 2]);
                swap(da[a[i]], da[a[(n + 1) / 2]]);
                swap(db[b[i]], db[b[(n + 1) / 2]]);
                if (i != (n + 1) / 2)
                {
                    ans.pb(make_pair(i, (n + 1) / 2));
                }
            }
            else
            {
                cout << -1 << endl;
                return;
            }
            ok = 0;
        }
        int temp = db[a[i]];
        swap(da[a[temp]], da[a[n - i + 1]]);
        swap(a[temp], a[n - i + 1]);
        swap(db[b[temp]], db[b[n - i + 1]]);
        swap(b[temp], b[n - i + 1]);
        if (temp != n - i + 1)
        {
            ans.pb(make_pair(temp, n - i + 1));
        }
    }
    ff(i, 1, n)
    {
        if (a[i] != b[n - i + 1])
        {
            cout << -1 << endl;
            return;
        }
    }
    cout << ans.size() << endl;
    for (auto x : ans)
    {
        cout << x.fi << " " << x.se << endl;
    }
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