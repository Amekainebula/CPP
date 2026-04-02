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
string a[10] = {"jia", "yi", "bing", "ding", "wu", "ji", "geng", "xin", "ren", "gui"};
string b[12] = {"zi", "chou", "yin", "mao", "chen", "si", "wu", "wei", "shen", "you", "xu", "hai"};
int p = 1984;
map<string, int> mp;
void pre()
{
    int tt = 60;
    int i = 0, j = 0;
    while (tt--)
    {
        if (i == 10)
            i = 0;
        if (j == 12)
            j = 0;
        mp[a[i] + b[j]] = p++;
        i++;
        j++;
    }
}
void solve()
{
    string t;
    cin >> t;
    cout << mp[t] << endl;
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int T = 1;
    cin >> T;
    pre();
    while (T--)
    {
        solve();
    }
    return 0;
}