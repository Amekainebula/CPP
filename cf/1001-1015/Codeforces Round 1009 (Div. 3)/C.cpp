#include <bits/stdc++.h>
// Finish Time: 2025/3/12 13:59:42 AC
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
void solve()
{
    int x;
    cin >> x;
    if (x == 2 || x == 3 || x == 4)
    {
        cout << -1 << endl;
        return;
    }

    bool flag = false;
    int ans = 0;
    int temp = x;
    int cnt = 0;
    int cnt0 = 0, cnt1 = 0;
    while (temp > 1)
    {
        if (!flag)
        {
            if (temp % 2 == 0)
            {
                ans += 1 << cnt;
                cnt0++;
            }
            else
            {
                flag = true;
                ans += 1 << cnt;
                cnt1++;
            }
        }
        else
        {
            if (temp % 2 == 0)
            {
                ans += 1 << cnt;
                cnt0++;
            }
            else
                cnt1++;
        }
        temp /= 2;
        cnt++;
    }
    if(cnt0 == 0 || cnt1 == 0)
    {
        cout << -1 << endl;
        return;
    }
    auto check = [&](int x, int y, int z)
    {
        return x + y > z && x + z > y && y + z > x;
    };
    if (flag)
    {
        // if(check(x,ans,x^ans))
        cout << ans << endl;
        // cout << ans << endl;
    }
    else
    {
        cout << -1 << endl;
    }
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int T = 1;
    cin >> T;
    while (T--)
    {
        solve();
    }
    return 0;
}