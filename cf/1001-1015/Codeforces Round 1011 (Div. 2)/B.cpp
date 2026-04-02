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
    vector<int> a(4);
    map<int, int> mp;
    vector<array<int, 2>> ans;
    if (n != 4)
    {
        cin >> a[0] >> a[1] >> a[2];

        for (int i = 3; i < n; i++)
        {
            int x;
            cin >> x;
            mp[x]++;
        }
        // int cnt = 0;
        for (int i = 0; i <= 2 * n; i++)
        {
            if (mp[i] == 0)
            {
                a[3] = i;
                break;
            }
        }
        ans.pb({4, n});
    }
    else
    {
        cin >> a[0] >> a[1] >> a[2] >> a[3];
    }
    int cnt0 = 0;
    for (int i = 0; i < 4; i++)
    {
        if (a[i] == 0)
            cnt0++;
    }
    if (cnt0 == 0)
    {
        ans.pb({1, 4});
    }
    else if (cnt0 == 1)
    {
        if (a[0] == 0 || a[1] == 0 || a[2] == 0)
            ans.pb({1, 3});
        else
            ans.pb({2, 4});
        ans.pb({1, 2});
    }
    else if (cnt0 == 2)
    {
        if (a[0] == 0 && a[1] == 0)
        {
            ans.pb({1, 2});
            ans.pb({1, 3});
        }
        else if (a[3] == 0 && a[2] == 0)
        {
            ans.pb({3, 4});
            ans.pb({1, 3});
        }
        else
        {
            ans.pb({1, 2});
            ans.pb({2, 3});
            ans.pb({1, 2});
        }
    }
    else
    {
        ans.pb({1, 2});
        ans.pb({2, 3});
        ans.pb({1, 2});
    }
    cout << ans.size() << endl;
    for (auto x : ans)
    {
        cout << x[0] << " " << x[1] << endl;
    }
    cout << a[0] << " " << a[1] << " " << a[2] << " " << a[3] << endl;
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int _T = 1;
    cin >> _T;
    while (_T--)
    {
        solve();
    }
    return 0;
}