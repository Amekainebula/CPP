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
//#define endl '\n'
#define WA AC
#define TLE AC
#define MLE AC
#define RE AC
#define CE AC
using namespace std;
const string AC = "Accepted";
vector<int> a(2, 0);
void solve()
{
    vector<int> ans(2, 0);
    ff(i, 0, 1)
    {
        cout << a[i] << endl;
        cin >> ans[i];
        ans[i] -= 2LL * a[i];
    }
    int x = 0, y = 0;
    ff(i, 0, 1)
    {
        for (int j = 29 - i; j >= 0; j -= 2)
        {
            int temp = ans[i] / (1 << j);
            if (temp == 2)
            {
                x += (1LL << j);
                y += (1LL << j);
            }
            else if (temp == 1)
                x += (1LL << j);
            ans[i] %= (1LL << j);
        }
    }
    cout << "!" << endl;
    int m;
    cin >> m;
    cout << (x | m) + (y | m) << endl;
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int T = 1;
    cin >> T;
    ffg(i, 0, 29) 
        a[i % 2] += (1LL << i);
    while (T--)
    {
        solve();
    }
    return 0;
}