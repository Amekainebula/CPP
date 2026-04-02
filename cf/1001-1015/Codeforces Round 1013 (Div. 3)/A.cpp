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
    vector<int> a(n+20, 0);
    bool ok = 0;
    ff(i, 1, n)
    {
        int x;
        cin >> x;
        a[x]+=1;
        if (a[0] >= 3 && a[1] >= 1 && a[2] >= 2 && a[3] >= 1 && a[5] >= 1 && !ok)
        {
            cout << i << endl;
            ok = 1;
            // break;
        }
    }
    if (!ok)
        cout << 0 << endl;
    // 01032025
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